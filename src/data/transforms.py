"""log -> z-score normalisation and its inverse (§3.5).

Concentrations span decades, so `u` and `s` are shifted-log transformed before
standardisation. The increment inverse is *not* the same map (see
`denormalize_delta`): the log Jacobian varies inside a coarse cell, which is why
the model projects the increment on the *physical* side.
"""
from __future__ import annotations

import json

import numpy as np
import torch

DEFAULT_LOG_FIELDS = ("u", "s")


class Normalizer:
    def __init__(self, mean: dict, std: dict, log_fields=DEFAULT_LOG_FIELDS) -> None:
        self.mean = {k: np.asarray(v, dtype=np.float64) for k, v in mean.items()}
        self.std = {k: np.maximum(np.asarray(v, dtype=np.float64), 1e-6) for k, v in std.items()}
        self.log_fields = set(log_fields)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, dataset, n_max: int = 100, log_fields=DEFAULT_LOG_FIELDS) -> "Normalizer":
        """Per-channel mean/std over `n_max` **strided** training samples.

        Strided, not `range(n)`: the dataset index is ordered by `(trajectory,
        step)`, so the first 100 windows are the first 100 steps of one
        trajectory -- mostly the pre-injection state. Fitted that way the `u`
        z-score comes out with `std ~ 1e-3`, every input is scaled by ~1e3 and
        the gradient of the last layer of a flux MLP reaches `1e4` on a `1e-5`
        loss. Striding covers the whole split, including the developed plume.
        """
        sums: dict[str, np.ndarray] = {}
        sqs: dict[str, np.ndarray] = {}
        n = min(len(dataset), n_max)
        if n == 0:
            raise ValueError("cannot fit a Normalizer on an empty dataset")
        picks = np.unique(np.linspace(0, len(dataset) - 1, n).astype(int))
        count = 0
        for i in picks:
            sample = dataset[i][0]
            # a Dataset sample is [C,N,N]; promote to [1,C,N,N] so the channel
            # axis is always dim 1 and the pooling axes are (0, 2, 3)
            for key in ("u", "q", "D", "s"):
                if key not in sample:
                    continue
                x = np.asarray(sample[key].detach().cpu().numpy(), dtype=np.float64)
                x = x[None] if x.ndim == 3 else x
                x = np.log1p(np.maximum(x, 0.0)) if key in log_fields else x
                axes = (0,) + tuple(range(2, x.ndim))
                csum = np.sum(x, axis=axes)
                csq = np.sum(x ** 2, axis=axes)
                sums[key] = csum if key not in sums else sums[key] + csum
                sqs[key] = csq if key not in sqs else sqs[key] + csq
            u = sample["u"]
            u = u[None] if u.ndim == 3 else u
            count += int(u.shape[0] * np.prod(u.shape[2:]))
        mean, std = {}, {}
        for key in sums:
            m = sums[key] / max(count, 1)
            v = sqs[key] / max(count, 1) - m ** 2
            mean[key] = m
            std[key] = np.sqrt(np.maximum(v, 0.0))
        return cls(mean, std, log_fields)

    # -------------------------------------------------------------- forward
    def _m(self, key: str, like: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean[key], dtype=like.dtype, device=like.device)
        return m.view(1, -1, 1, 1)

    def _s(self, key: str, like: torch.Tensor) -> torch.Tensor:
        s = torch.as_tensor(self.std[key], dtype=like.dtype, device=like.device)
        return s.view(1, -1, 1, 1)

    def normalize(self, key: str, x: torch.Tensor) -> torch.Tensor:
        z = torch.log1p(x.clamp_min(0.0)) if key in self.log_fields else x
        return (z - self._m(key, x)) / self._s(key, x)

    def denormalize(self, key: str, z: torch.Tensor) -> torch.Tensor:
        v = z * self._s(key, z) + self._m(key, z)
        return torch.expm1(v).clamp_min(0.0) if key in self.log_fields else v

    # convenience aliases used by the model code
    def normalize_u(self, u: torch.Tensor) -> torch.Tensor:
        return self.normalize("u", u)

    def denormalize_u(self, z: torch.Tensor) -> torch.Tensor:
        return self.denormalize("u", z)

    def normalize_params(self, key: str, x: torch.Tensor) -> torch.Tensor:
        """z-score for q / D / s; identity for the five physical coefficients."""
        if key not in self.mean:
            return x
        return self.normalize(key, x)

    def denormalize_delta(self, dz: torch.Tensor, u_ref: torch.Tensor, key: str = "u") -> torch.Tensor:
        """Physical increment from a z-space increment.

        u = expm1(z*s + m) => du = (u + 1) * s * dz. Using `denormalize_u(dz)`
        instead would be a factor-u error, largest exactly at the front.
        """
        return dz * self._s(key, dz) * (u_ref.clamp_min(0.0) + 1.0)

    # ------------------------------------------------------------ (de)serialise
    def state_dict(self) -> dict:
        out = {f"mean_{k}": v.astype(np.float32) for k, v in self.mean.items()}
        out.update({f"std_{k}": v.astype(np.float32) for k, v in self.std.items()})
        out["log_fields"] = np.array(sorted(self.log_fields), dtype="U8")
        return out

    def save(self, path: str, provenance: dict | None = None) -> None:
        """`provenance` is stamped into the file so a stale `stats.npz` can be spotted.

        Without it a stats file frozen from one dataset silently normalises another
        (regenerated bands, a different `n_max`), and every arm inherits the error.
        """
        state = self.state_dict()
        if provenance:
            state["provenance"] = np.array(json.dumps(provenance, sort_keys=True, default=str))
        np.savez(path, **state)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        with np.load(path, allow_pickle=False) as fh:
            keys = set(fh.files)
            mean = {k[5:]: fh[k] for k in keys if k.startswith("mean_")}
            std = {k[4:]: fh[k] for k in keys if k.startswith("std_")}
            log_fields = tuple(fh["log_fields"].tolist()) if "log_fields" in keys else DEFAULT_LOG_FIELDS
        return cls(mean, std, log_fields)

    @classmethod
    def provenance(cls, path: str) -> dict:
        with np.load(path, allow_pickle=False) as fh:
            if "provenance" not in fh.files:
                return {}
            return json.loads(str(fh["provenance"]))

    def to_state(self) -> dict:
        return {"mean": {k: v.tolist() for k, v in self.mean.items()},
                "std": {k: v.tolist() for k, v in self.std.items()},
                "log_fields": sorted(self.log_fields)}

    @classmethod
    def from_state(cls, state: dict) -> "Normalizer":
        return cls(state["mean"], state["std"], tuple(state.get("log_fields", DEFAULT_LOG_FIELDS)))
