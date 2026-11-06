"""Conservation-Exact Multiscale Operator (implementation.md §3.8).

One step is three moves, in this order and no other:

1. **Coarse flux.** The coarse cell means are handed to a spectral stack that
   emits face fluxes; `divergence()` telescopes them, so their grid sum is zero
   to float rounding.
2. **Progressive detail.** The fine-scale residual `u - Pi u` is gated by its own
   wavelet detail energy and re-injected through a fine-grid conv, with a block
   mean of exactly zero.
3. **Analytic source/decay.** Applied last, in composite variables.

Design deviation recorded in `docs/dev_log.md` (2026-10-30): the transport
increment is applied in the **composite** variable `m = theta C + rho_b S(C)` and
the detail branch is bounded by the local mass and projected in `m`. With a
nonlinear isotherm a zero-mean increment in `C` is *not* a zero-mean increment in
`m`, so a `C`-space projection would leave an O(S'') residual in the very budget
the paper audits. Everything downstream (`total_mass`, `audit_series`) is
unchanged; only the space the increment lives in changed, and it changed to the
one the claim is stated in.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .balance import analytic_step, composite_mass, langmuir_inverse
from .blocks import (CoarseFluxPropagator, MassPartition, SoftGate,
                     WaveletResidualBranch, divergence, count_params,
                     mass_weighted_project, positivity_limiter)
from ..utils.config import get


class CMO(nn.Module):
    def __init__(self, cfg: Any, normalizer: Any = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_fine = int(get(cfg, "data.N", 64))
        self.block = int(get(cfg, "model.block", 2))
        self.use_separation = bool(get(cfg, "model.use_separation", True))
        self.use_projection = bool(get(cfg, "model.use_projection", True))
        self.use_gate = bool(get(cfg, "model.gate", True))
        self.debug = bool(get(cfg, "run.debug", False))
        self.dx = float(get(cfg, "data.dx", 1.0))
        if self.n_fine % self.block:
            raise ValueError(f"data.N={self.n_fine} not divisible by model.block={self.block}")

        self.partition = MassPartition(self.n_fine, self.block)
        self.coarse = CoarseFluxPropagator(
            in_ch=CoarseFluxPropagator.IN_CH,
            width=int(get(cfg, "model.coarse_width", 16)),
            modes=int(get(cfg, "model.modes", 12)),
            n_layers=int(get(cfg, "model.n_layers", 4)),
        )
        if self.use_separation:
            self.wave = WaveletResidualBranch(
                width=int(get(cfg, "model.wave_width", 32)),
                in_ch=WaveletResidualBranch.IN_CH,
                wavelet=str(get(cfg, "model.wavelet", "db2")),
                levels=int(get(cfg, "model.levels", 2)),
            )
            self.gate = SoftGate(float(get(cfg, "model.gate.beta_init", 4.0)),
                                 float(get(cfg, "model.gate.tau_init", 0.1)))
        # one positive scale per face direction: a flux is a unit-bearing object
        # and a per-channel scale would silently absorb the divergence structure
        # `flux_form` is the 2x2 cell of the baseline comparison (dev_log 2026-11-01):
        #   "relative" (default) -- CMO/NFV form: unbounded head x cell mean x learned
        #                           scale, then dt * div(F)/dx.
        #   "absolute"           -- FluxNet form: bounded face *amount* tanh(F) * 0.5 *
        #                           upwind mass, dt/dx absorbed into the bound. Same
        #                           separation, different flux parametrisation.
        self.flux_form = str(get(cfg, "model.flux_form", "relative")).lower()
        if self.flux_form not in ("relative", "absolute"):
            raise ValueError(f"model.flux_form must be 'relative' or 'absolute', got "
                             f"'{self.flux_form}'")
        if self.flux_form == "relative":
            self.log_flux_scale = nn.Parameter(torch.zeros(2))
        self.normalizer = normalizer

    # ---------------------------------------------------------------- checks
    def _check_batch(self, batch: dict) -> None:
        u = batch["u"]
        if u.dim() != 4 or u.shape[1] != 1:
            raise ValueError(f"batch['u'] must be [B,1,N,N], got {tuple(u.shape)}")
        N = u.shape[-1]
        for key, ch in (("q", 2), ("D", 3), ("s", 1)):
            v = batch[key]
            if tuple(v.shape) != (u.shape[0], ch, N, N):
                raise ValueError(
                    f"batch['{key}'] must be {(u.shape[0], ch, N, N)}, got {tuple(v.shape)}"
                )
        for key in ("theta", "rho_b", "smax", "k_lang", "lam"):
            if key not in batch:
                raise KeyError(f"batch['{key}'] missing (scalar coefficient)")
            if batch[key].numel() != u.shape[0]:
                raise ValueError(
                    f"batch['{key}'] must hold one scalar per sample, got shape {tuple(batch[key].shape)}"
                )
        if not float(batch["dt"]) > 0:
            raise ValueError(f"batch['dt'] must be positive, got {batch['dt']}")

    def _partition_for(self, n: int) -> MassPartition:
        """Rebuild Pi when the grid changes (128^2 transfer); it has no weights."""
        if n != self.partition.m * self.partition.M:
            self.partition = MassPartition(n, self.block)
        return self.partition

    # -------------------------------------------------------------- machinery
    def _params_c(self, batch: dict, part: MassPartition) -> dict[str, torch.Tensor]:
        norm = self.normalizer
        out = {}
        for key in ("q", "D", "s"):
            v = part.restrict(batch[key])
            out[key] = norm.normalize_params(key, v) if norm is not None else v
        for key in ("theta", "rho_b", "smax", "k_lang", "lam"):
            out[key] = batch[key].reshape(-1, 1, 1, 1)
        return out

    def _scalars(self, batch: dict) -> dict[str, torch.Tensor]:
        return {k: batch[k].reshape(-1, 1, 1, 1)
                for k in ("theta", "rho_b", "smax", "k_lang", "lam")}

    def forward(self, batch: dict, return_diag: bool = False):
        self._check_batch(batch)
        u = batch["u"]
        dt = float(batch["dt"])
        N = u.shape[-1]
        part = self._partition_for(N)
        sc = self._scalars(batch)
        # coefficients stay [B,1,1,1] tensors: samples in a batch have different
        # theta / rho_b / smax / k_lang, and collapsing them to a float would
        # make the inverse inconsistent with the audit for every sample but #0
        theta, rho_b, smax, k_lang, lam = (sc[k] for k in
                                           ("theta", "rho_b", "smax", "k_lang", "lam"))

        # --- 1. coarse flux -------------------------------------------------
        u_c = part.restrict(u)
        params_c = self._params_c(batch, part)
        norm = self.normalizer
        u_c_z = norm.normalize_u(u_c) if norm is not None else u_c
        Fhat, feat = self.coarse(u_c_z, params_c)
        m_now = composite_mass(u, theta, rho_b, smax, k_lang)
        if self.flux_form == "absolute":
            # FluxNet's parametrisation on the coarse grid (dev_log 2026-11-01):
            # a bounded *amount* per face, at most half the upwind cell mass, and
            # no dt/dx factor -- the bound absorbs them, exactly as in fluxnet.py.
            #
            # `_bound_time_scale` exists because that absorption makes the branch
            # **dt-blind**: the relative form advances by `dt * div(F)` while this one
            # advances by a fixed fraction of the upwind mass per call, so the same
            # weights move 6.25x more mass per unit time when dt shrinks 6.25x
            # (`adr2d_ext.h5`, dt=0.016 vs 0.1). Scaling the bound by `dt/dt_train`
            # is the minimal repair, and exposing it as an attribute keeps the
            # default (`1.0`) bit-identical to what training and every published
            # number used. Set it only from an analysis script, never in a config:
            # a trainable/configured value here would silently redefine the baseline.
            m_c = part.restrict(m_now)
            bound = 0.5 * float(getattr(self, "_bound_time_scale", 1.0))
            tx = torch.tanh(Fhat[:, 0:1]) * bound * torch.roll(m_c, 1, dims=-1).clamp_min(0.0)
            ty = torch.tanh(Fhat[:, 1:2]) * bound * torch.roll(m_c, 1, dims=-2).clamp_min(0.0)
            dm_c = -((tx - torch.roll(tx, -1, dims=-1))
                     + (ty - torch.roll(ty, -1, dims=-2)))
        else:
            scale = F.softplus(self.log_flux_scale).view(1, 2, 1, 1)
            # A flux is carried by solute: multiplying by the local cell mean makes
            # the network predict a *relative* flux (q*C-like), so an untrained net at
            # u=0 predicts u=0 instead of injecting mass where there is none. Without
            # this the positivity limiter has to clamp the whole field to zero and the
            # flux branch never sees a gradient.
            Fhat_phys = Fhat * scale * u_c
            flux_div = divergence(Fhat_phys, self.dx)          # sums to zero
            dm_c = -dt * flux_div

        # --- 2a. transport in the composite variable ------------------------
        # lift(dm) sums to zero because dm_c does; the limiter scales that
        # zero-sum field, so it stays zero-sum while guaranteeing m >= 0
        dm = positivity_limiter(part.lift(dm_c), m_now)
        m_new = m_now + dm
        C_new, n_clamped = langmuir_inverse(m_new, theta, rho_b, smax, k_lang)

        # --- 2b. progressive detail (bounded, projected in m) ---------------
        gate_mean = torch.zeros((), device=u.device)
        Ehat = None
        if self.use_separation:
            # pseudo-inverse of the log-space Jacobian: du = (u+1) * s * dz
            u_ref = u.clamp_min(0.0)
            du_scale = (u_ref + 1.0) * self._u_std(u)
            r_z = (u - part.forward(u)) / du_scale
            q_z = norm.normalize_params("q", batch["q"]) if norm is not None else batch["q"]
            D_z = norm.normalize_params("D", batch["D"]) if norm is not None else batch["D"]
            beta = self.gate.beta if self.use_gate else torch.tensor(4.0, device=u.device)
            tau = self.gate.tau if self.use_gate else torch.tensor(0.1, device=u.device)
            raw, g, Ehat = self.wave(r_z, q_z, D_z, beta, tau)
            if self.use_gate:
                g_up = part.lift(part.restrict(F.interpolate(g, size=(N, N), mode="nearest")))
                gate_mean = g.detach().mean()
            else:
                g_up = torch.ones_like(raw)
            # Rate bound, not a mass bound: fine-scale transport happens at the
            # velocity scale, so the detail may move at most `cfl` times the
            # local mass per step. `dt * m` alone would allow a 20% jump per step
            # at dt=0.2, which an untrained net turns into positive feedback.
            qmag = batch["q"].pow(2).sum(1, keepdim=True).sqrt()
            cfl = (dt * qmag / self.dx).clamp(0.0, 0.25)
            rate = cfl * g_up * torch.tanh(raw)
            if self.use_projection:
                delta_m = mass_weighted_project(rate, m_new, part)
            else:
                delta_m = m_new * rate
            delta_m = positivity_limiter(delta_m, m_new)
            m_final = m_new + delta_m
            C_new, extra = langmuir_inverse(m_final, theta, rho_b, smax, k_lang)
            n_clamped += extra

        # --- 3. analytic source + decay ------------------------------------
        C_out, extra = analytic_step(C_new, torch.zeros_like(C_new), batch["s"],
                                     lam, theta, rho_b, smax, k_lang, dt)
        n_clamped += extra

        if not return_diag:
            return C_out
        u2 = part.residual(C_out)
        with torch.no_grad():
            rho = (u2 ** 2).sum() / (C_out ** 2).sum().clamp_min(1e-30)
            # the absolute form has no learned scale: its bound is the upwind mass
            scales = ([float(v) for v in F.softplus(self.log_flux_scale.detach())]
                      if self.flux_form == "relative" else [0.0, 0.0])
        return C_out, {
            "rho": float(rho),
            "gate_mean": float(gate_mean),
            "clamp_count": int(n_clamped),
            "flux_scale": scales,
            "detail_energy": None if Ehat is None else float(Ehat.detach().mean()),
        }

    def _u_std(self, u: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            return torch.ones_like(u)
        s = torch.as_tensor(self.normalizer.std["u"], dtype=u.dtype, device=u.device)
        return s.reshape(1, -1, 1, 1)

    def n_params(self) -> int:
        return count_params(self)
