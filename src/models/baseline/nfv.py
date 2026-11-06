"""NFV baseline (implementation.md §3.9.3) — the control that isolates separation.

Same divergence form as `CMO`: a learned numerical flux on faces, a telescoping
divergence, a zero-sum increment, the same analytic source. The one thing
removed is the **scale separation** — no block averaging, no spectral coarse
solve, no wavelet branch, no gate. So:

* `NFV` conserves mass to the same order as `CMO` (the audit should say so);
* if `CMO` still wins on front sharpness, the difference cannot be conservation
  and cannot be the source term: it is the separation.

The flux is parameterised as a *relative* flux times the upwind composite mass,
so `m = 0` gives `F = 0` and an untrained net cannot invent mass.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..balance import analytic_step, composite_mass, langmuir_inverse
from ..blocks import count_params, detail_ratio, divergence, positivity_limiter
from ...utils.config import get


class NFVBaseline(nn.Module):
    IN = 4  # m_L, m_R, q.n, D.n

    def __init__(self, cfg: Any, normalizer: Any = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.normalizer = normalizer
        self.dx = float(get(cfg, "data.dx", 1.0))
        # `model.face_width`, not `model.coarse_width`: the latter already means
        # the spectral width of CMO's coarse propagator (16 in default.yaml), so
        # widening the baselines through it would have widened CMO instead. The
        # fallback chain keeps every pre-existing yaml buildable.
        width = int(get(cfg, "model.face_width", get(cfg, "model.coarse_width", 64)))
        self.mlp = nn.Sequential(
            nn.Linear(self.IN, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # the explicit `dx` of a finite-volume flux (CMO carries a learned scale
        # instead); reported in the parameter table so the comparison stays honest
        self.log_flux_scale = nn.Parameter(torch.zeros(2))

    def _norm(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return x if self.normalizer is None else self.normalizer.normalize_params(key, x)

    def _flux(self, m: torch.Tensor, q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """[B,2,N,N] face fluxes with index i = the face between i-1 and i.

        Normalization happens per component *before* selection: the `q`/`D` stats
        hold two entries each, so normalizing a single component would broadcast
        it into two channels and feed the MLP 7 inputs instead of 4.
        """
        qz, Dz = self._norm("q", q), self._norm("D", D)
        m_n = self._norm("u", m)
        chans = []
        for dim, comp in ((-1, 0), (-2, 1)):
            m_l = torch.roll(m_n, 1, dims=dim)
            qc, Dc = qz.select(1, comp), Dz.select(1, comp)
            qn = 0.5 * (qc + torch.roll(qc, 1, dims=dim))
            Dn = 0.5 * (Dc + torch.roll(Dc, 1, dims=dim))
            h = torch.cat([m_l, m_n, qn.unsqueeze(1), Dn.unsqueeze(1)], dim=1)
            rel = self.mlp(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            chans.append(rel * m_l.clamp_min(0.0))
        Fhat = torch.cat(chans, dim=1)
        return Fhat * F.softplus(self.log_flux_scale).view(1, 2, 1, 1)

    def forward(self, batch: dict, return_diag: bool = False):
        u = batch["u"]
        if u.dim() != 4 or u.shape[1] != 1:
            raise ValueError(f"batch['u'] must be [B,1,N,N], got {tuple(u.shape)}")
        sc = {k: batch[k].reshape(-1, 1, 1, 1)
              for k in ("theta", "rho_b", "smax", "k_lang", "lam")}
        theta, rho_b, smax, k_lang = sc["theta"], sc["rho_b"], sc["smax"], sc["k_lang"]

        m = composite_mass(u, theta, rho_b, smax, k_lang)
        flux_div = divergence(self._flux(m, batch["q"], batch["D"]), self.dx)
        dm = positivity_limiter(-float(batch["dt"]) * flux_div, m)
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
            "flux_scale": [float(v) for v in F.softplus(self.log_flux_scale)],
            "detail_energy": 0.0,
            "mass_defect": float(dm.sum()),
        }

    def n_params(self) -> int:
        return count_params(self)
