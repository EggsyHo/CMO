"""FNO2d baseline (implementation.md §3.9.1) — the "fast but leaky" arm.

The network predicts the **increment**, not the state. Predicting `u^{n+1}`
directly is the weaker baseline: any drift in the loss would be blamed on the
parameterisation instead of on the missing conservation. With an increment the
baseline is strictly stronger, so whatever drift it shows is attributable to
`sum(du) != 0`.

The source/decay term is the *same* `analytic_step` the operator uses, so the
only difference between `FNO` and `CMO` is how transport is represented.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..balance import analytic_step, composite_mass
from ..blocks import SpectralConv2d, count_params, detail_ratio
from ...utils.config import get


class FNO2dBaseline(nn.Module):
    # u(1) + q(2) + D(3: xx, yy, xy) + s(1) -- exactly the seven channels of
    # implementation.md §3.9.1, i.e. every field the operator also sees
    IN_CH = 7

    def __init__(self, cfg: Any, normalizer: Any = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.normalizer = normalizer
        self.width = int(get(cfg, "model.coarse_width", 32))
        self.modes = int(get(cfg, "model.modes", 12))
        self.n_layers = int(get(cfg, "model.n_layers", 4))

        self.fc0 = nn.Conv2d(self.IN_CH, self.width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes) for _ in range(self.n_layers)]
        )
        self.bypass = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)]
        )
        self.fc1 = nn.Conv2d(self.width, 128, 1)
        self.fc2 = nn.Conv2d(128, 1, 1)
        # Zero output layer. With `std=0.02` the untrained arm injects 76% of the
        # domain's mass in a *single* step (measured), which is not "drift": it is
        # an untrainable initial condition. At zero the arm starts as pure
        # source/decay and every gram of drift it later shows is fitted, not an
        # init artefact -- which is what the comparison needs.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    # ------------------------------------------------------------------ input
    def _features(self, batch: dict) -> torch.Tensor:
        u, q, D, s = batch["u"], batch["q"], batch["D"], batch["s"]
        norm = self.normalizer
        if norm is None:
            uz, qz, Dz, sz = u, q, D, s
        else:
            uz = norm.normalize_u(u)
            qz = norm.normalize_params("q", q)
            Dz = norm.normalize_params("D", D)
            sz = norm.normalize_params("s", s)
        return torch.cat([uz, qz, Dz, sz], dim=1)

    def _scalars(self, batch: dict) -> dict[str, torch.Tensor]:
        return {k: batch[k].reshape(-1, 1, 1, 1)
                for k in ("theta", "rho_b", "smax", "k_lang", "lam")}

    # ---------------------------------------------------------------- forward
    def forward(self, batch: dict, return_diag: bool = False):
        u = batch["u"]
        if u.dim() != 4 or u.shape[1] != 1:
            raise ValueError(f"batch['u'] must be [B,1,N,N], got {tuple(u.shape)}")
        sc = self._scalars(batch)

        z = self.fc0(self._features(batch))
        for sp, bp in zip(self.spectral, self.bypass):
            z = F.gelu(sp(z) + bp(z))
        du_z = self.fc2(F.gelu(self.fc1(z)))

        if self.normalizer is None:
            du = du_z
        else:
            du = self.normalizer.denormalize_delta(du_z, u)
        u_adv = u + du                      # no projection: sum(du) != 0 in general

        C_out, n_clamped = analytic_step(u_adv, torch.zeros_like(u_adv), batch["s"],
                                        sc["lam"], sc["theta"], sc["rho_b"],
                                        sc["smax"], sc["k_lang"], float(batch["dt"]))
        if not return_diag:
            return C_out
        return C_out, {
            "rho": detail_ratio(C_out, int(get(self.cfg, "model.block", 2))),
            "gate_mean": 0.0,
            "clamp_count": int(n_clamped),
            "flux_scale": [0.0, 0.0],
            "detail_energy": 0.0,
            "mass_defect": float(du.mean()),
        }

    def n_params(self) -> int:
        return count_params(self)
