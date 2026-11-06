"""FluxNet baseline (implementation.md §3.9.2) — "exact but blunt".

A single shared MLP is evaluated on every face of the fine grid and returns the
**amount of mass to move across that face**. The update is written as
`m_i <- m_i - (outflow_i - inflow_i)`, and each face enters exactly two cells
with opposite signs, so total mass is conserved by construction — this arm is
the control for "conservation alone is not enough".

What it does *not* have: any coupling beyond one face (no coarse/global
transport), no separation between resolved and sub-grid motion, no gate. Local
stencils with learned weights cannot carry a plume faster than one cell per
layer, which is precisely the failure the operator is designed to avoid.

The capacity clamp `|T| <= 0.5 * m_L` (implementation.md §0.4) keeps one step
from draining a cell, so the scheme stays usable at the training time step.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..balance import analytic_step, composite_mass, langmuir_inverse
from ..blocks import count_params, detail_ratio, positivity_limiter
from ...utils.config import get


class FluxNetBaseline(nn.Module):
    IN = 4  # m_L, m_R, q.n, D.n

    def __init__(self, cfg: Any, normalizer: Any = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.normalizer = normalizer
        # see nfv.py: `coarse_width` is CMO's spectral width, `face_width` is this
        # arm's per-face MLP width. Conflating them made the parameter comparison
        # unfair *against* the baselines at 1/68th of CMO's count.
        width = int(get(cfg, "model.face_width", get(cfg, "model.coarse_width", 64)))
        self.mlp = nn.Sequential(
            nn.Linear(self.IN, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 1),
        )
        # zero output: at init the arm reduces to pure source/decay, so the
        # transport it later learns is fitted, never inherited from an init scale
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    # ---------------------------------------------------------------- features
    def _norm(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return x if self.normalizer is None else self.normalizer.normalize_params(key, x)

    def _amount(self, m_l: torch.Tensor, m_r: torch.Tensor,
                qn: torch.Tensor, Dn: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self._norm("u", m_l), self._norm("u", m_r), qn, Dn], dim=1)
        raw = self.mlp(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return torch.tanh(raw) * 0.5 * m_l.clamp_min(0.0)

    def _face_terms(self, m: torch.Tensor, q: torch.Tensor, D: torch.Tensor):
        """Face amounts with index i = the face between cell i-1 and cell i.

        `q`/`D` are normalized *before* the component is selected: their statistics
        are per-component, so normalizing one component against the 2-channel
        stats silently broadcasts a `[B,1,N,N]` tensor into `[B,2,N,N]`.
        """
        qz, Dz = self._norm("q", q), self._norm("D", D)
        terms = []
        for dim, comp in ((-1, 0), (-2, 1)):
            m_l = torch.roll(m, 1, dims=dim)
            qc, Dc = qz.select(1, comp), Dz.select(1, comp)
            qn = 0.5 * (qc + torch.roll(qc, 1, dims=dim))
            Dn = 0.5 * (Dc + torch.roll(Dc, 1, dims=dim))
            terms.append(self._amount(m_l, m, qn.unsqueeze(1), Dn.unsqueeze(1)))
        return terms

    # ---------------------------------------------------------------- forward
    def forward(self, batch: dict, return_diag: bool = False):
        u = batch["u"]
        if u.dim() != 4 or u.shape[1] != 1:
            raise ValueError(f"batch['u'] must be [B,1,N,N], got {tuple(u.shape)}")
        sc = {k: batch[k].reshape(-1, 1, 1, 1)
              for k in ("theta", "rho_b", "smax", "k_lang", "lam")}
        theta, rho_b, smax, k_lang = sc["theta"], sc["rho_b"], sc["smax"], sc["k_lang"]

        m = composite_mass(u, theta, rho_b, smax, k_lang)
        T_x, T_y = self._face_terms(m, batch["q"], batch["D"])
        # cell i loses the amount on its own face (i) and gains the one on face
        # (i+1); both faces appear once with each sign, hence exact conservation
        dm = -((T_x - torch.roll(T_x, -1, dims=-1)) + (T_y - torch.roll(T_y, -1, dims=-2)))
        dm = positivity_limiter(dm, m)
        C_new, n_clamped = langmuir_inverse(m + dm, theta, rho_b, smax, k_lang)

        C_out, extra = analytic_step(C_new, torch.zeros_like(C_new), batch["s"],
                                     sc["lam"], theta, rho_b, smax, k_lang,
                                     float(batch["dt"]))
        n_clamped += extra
        if not return_diag:
            return C_out
        return C_out, {
            "rho": detail_ratio(C_out, int(get(self.cfg, "model.block", 2))),
            "gate_mean": 0.0,
            "clamp_count": int(n_clamped),
            "flux_scale": [0.0, 0.0],
            "detail_energy": 0.0,
            "mass_defect": float(dm.sum()),
        }

    def n_params(self) -> int:
        return count_params(self)
