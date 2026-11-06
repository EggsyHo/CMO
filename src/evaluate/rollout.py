"""Free-running rollouts and the mass audit (implementation.md §3.14).

The audit is written in **numpy, from the definition**, not by calling the torch
isotherm. That is deliberate: `metrics.py` re-derives the partition in numpy for
the same reason, so a sign error inside `balance.py` cannot certify its own
correctness. If both implementations agree to machine precision, the residual is
the model's; if they disagree, the audit fails loudly instead of reporting 1e-16.
"""
from __future__ import annotations

import numpy as np
import torch

from ..models.balance import total_mass


@torch.no_grad()
def rollout(model, batch: dict, steps: int, dx: float = 1.0,
            device: torch.device | None = None) -> dict:
    """Free-running rollout; returns predictions, the mass series and diagnostics."""
    device = device or next(model.parameters()).device
    model.eval()
    dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    u = dev["u"]
    src_seq = dev.get("src_seq")
    sc = {k: dev[k] for k in ("theta", "rho_b", "smax", "k_lang")}
    steps = int(min(steps, src_seq.shape[1] if src_seq is not None else steps))

    preds = [u]
    mass = [total_mass(u, **sc, dx=dx).double().cpu().reshape(u.shape[0])]
    clamp, gate = [0], [0.0]
    for k in range(steps):
        src = src_seq[:, k] if src_seq is not None else dev["s"]
        out, diag = model(dict(dev, u=u, s=src), return_diag=True)
        u = out
        preds.append(u)
        mass.append(total_mass(u, **sc, dx=dx).double().cpu().reshape(u.shape[0]))
        clamp.append(int(diag.get("clamp_count", 0)))
        gate.append(float(diag.get("gate_mean") or 0.0))
    return {
        "pred": torch.stack(preds, 1).float().cpu(),        # [B, T+1, 1, N, N]
        "mass": torch.stack(mass, 1).numpy(),               # [B, T+1]
        "clamp": np.asarray(clamp, dtype=np.int64),         # [T+1]
        "gate": np.asarray(gate, dtype=np.float64),          # [T+1]
        "steps": steps,
    }


def _composite_mass_np(C: np.ndarray, theta, rho_b, smax, k_lang) -> np.ndarray:
    """`m = theta*C + rho_b*smax*K*C/(1+K*C)`, broadcast over [B,T,1,N,N]."""
    return theta * C + rho_b * smax * (k_lang * C) / (1.0 + k_lang * C)


def audit_series(pred: torch.Tensor, batch: dict, dx: float = 1.0,
                 budget: str = "composite") -> dict:
    """Mass residual per step against the analytic source/decay budget.

    `r_k = |M_k - M_0 - sum_{j<k} dt*(S_j - lam*M_j)| / M_0`, exactly the budget
    `implementation.md` Part 3 fixes in advance. Valid as a *test* only because
    the transport increment is zero-sum: the operator applies decay to the
    post-transport mass, and the two orderings agree in the total because the
    transport moves mass without creating it. An arm that leaks transport mass
    shows it here and nowhere else.
    """
    C = pred.detach().cpu().numpy().astype(np.float64)
    B, T = C.shape[0], C.shape[1]

    def scal(key):
        v = batch[key]
        v = v.detach().cpu().numpy().astype(np.float64).reshape(-1)
        return v.reshape(B, 1) if v.size == B else np.full((B, 1), float(v.reshape(-1)[0]))

    theta, rho_b, smax, k_lang = (scal(k) for k in ("theta", "rho_b", "smax", "k_lang"))
    lam = scal("lam")
    dt = float(batch["dt"])
    if budget == "composite":
        field = _composite_mass_np(C, theta[:, :, None, None, None], rho_b[:, :, None, None, None],
                                   smax[:, :, None, None, None], k_lang[:, :, None, None, None])
    elif budget == "aqueous":
        field = theta[:, :, None, None, None] * C
    else:
        raise ValueError(f"unknown budget '{budget}' (composite|aqueous)")

    M = field.sum(axis=(2, 3, 4)) * dx ** 2                       # [B, T]
    if "src_seq" in batch:
        S = batch["src_seq"].detach().cpu().numpy().astype(np.float64)
        S = S.sum(axis=(2, 3, 4)) * dx ** 2
        S = S[:, : T - 1]
    else:
        s0 = batch["s"].detach().cpu().numpy().astype(np.float64).sum(axis=(1, 2, 3)) * dx ** 2
        S = np.repeat(s0[:, None], T - 1, axis=1)

    cum = np.zeros((B, T))
    for k in range(1, T):
        cum[:, k] = cum[:, k - 1] + dt * (S[:, k - 1] - lam[:, 0] * M[:, k - 1])
    resid = M - M[:, :1] - cum
    # Relative to the initial mass. If the window starts *before* injection the
    # initial mass is exactly 0 and the ratio is 0/0 -- it must not be reported as
    # a 1e23 "violation". Such windows fall back to the trajectory peak and are
    # flagged, so a degenerate window is visible instead of silently rescaled.
    peak = np.maximum(np.abs(M).max(axis=1, keepdims=True), 1e-30)
    scale = np.abs(M[:, :1])
    degenerate = scale < 1e-12 * peak
    scale = np.where(degenerate, peak, scale).clip(min=1e-30)
    r = np.abs(resid) / scale
    return {"r": r, "M": M, "r_final": r[:, -1], "degenerate": degenerate.reshape(-1),
            "r_final_max": float(r[:, -1].max()), "r_all_max": float(r.max()),
            "budget": budget}
