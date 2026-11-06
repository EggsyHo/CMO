"""HDF5 -> (batch dict, target) on a fixed split (§3.3).

HDF5 layout written by `data/gen_adr2d.py`:

    /traj_0000/C    [T+1, N, N] float32   concentration C [mg/L]
    /traj_0000/src  [T, N, N]   float32   injection source [mg/L/s]
    /traj_0000/qx, qy [N, N]    float32   Darcy velocity (m/s)
    /traj_0000/D    [3, N, N]   float32   (Dxx, Dxy, Dyy) [m^2/s]
    attrs on /traj_0000: theta rho_b smax k_lang lam x0 y0 Pe Da kappa R0
    /params  [n, 9] float64, columns named by the `columns` attr
    /splits/{train,val,test} int arrays
    /meta attrs: N T dt dx regime seed generator_version

**Unit system** (self-consistent, and the one the generator solves in): C [mg/L],
S [mg/kg], rho_b [kg/L], K_L [L/mg], theta [-], lam [1/s], q [m/s], dx [m]. Then
`theta*C + rho_b*S(C)` is mg/L on both terms, which is what makes the composite
budget addable. The earlier "kg/m^3 / m^3/mg" labels in docs §2 belonged to a
unit system the generator does not use: with rho_b ~ 1.5 kg/L a kg/m^3 label
would silently rescale the sorbed phase by 1000. Nothing numerical changes by
fixing the label, which is exactly why the label had to be fixed.

Everything here stays in physical units: the audit must not measure a rescaled budget.
"""
from __future__ import annotations

import os
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import Normalizer

PARAM_COLUMNS = ("Pe", "Da", "kappa", "theta", "rho_b", "smax", "k_lang", "lam", "dt")


class ADRDataset(Dataset):
    def __init__(self, path: str, split: str = "train", K: int = 4,
                 stats_path: str | None = None, dtype=torch.float32) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"dataset not found: {path}")
        self.path = path
        self.split = split
        self.K = int(K)
        self.dtype = dtype
        self._fh: h5py.File | None = None

        with h5py.File(path, "r") as fh:
            if f"/splits/{split}" not in fh:
                raise KeyError(f"{path} has no /splits/{split}")
            self.split_ids = np.asarray(fh[f"/splits/{split}"][:], dtype=np.int64)
            meta = dict(fh["/meta"].attrs) if "/meta" in fh else {}
            self.regime = str(meta.get("regime", "unknown"))
            self.N = int(meta.get("N", 64))
            self.T = int(meta.get("T", 0))
            self.dx = float(meta.get("dx", 1.0))
            self.dt = float(meta.get("dt", 0.02))
            self.traj_names = sorted(k for k in fh.keys() if k.startswith("traj_"))

        if self.split_ids.size == 0:
            if split == "train":
                raise ValueError(
                    f"{path} has an empty train split (regime={self.regime}). "
                    "Zero-shot files must be loaded through ZeroShotDataset."
                )
            raise ValueError(f"split '{split}' is empty in {path}")

        # a trajectory is sampled at (traj, n) with a full window available:
        # n ranges over [0, T_i - K], which is T_i - K + 1 windows. `range(T_i - K)`
        # is one short and drops the *last* step of every trajectory -- the step a
        # long-horizon test wants most (§3.3).
        self.index: list[tuple[int, int]] = []
        with h5py.File(path, "r") as fh:
            for tid in self.split_ids:
                T_i = fh[f"/{self.traj_names[int(tid)]}/C"].shape[0] - 1
                self.index.extend((int(tid), n) for n in range(max(T_i - self.K + 1, 0)))
        self.n_traj = len(self.split_ids)
        self.T = int(self.T or max((t for _, t in self.index), default=0))

        self.params = self._read_params()
        if stats_path and os.path.exists(stats_path):
            self.normalizer = Normalizer.load(stats_path)
        else:
            self.normalizer = None

    def fit_normalizer(self, n_max: int = 100) -> Normalizer:
        """Fit on `n_max` strided samples of this split (call on the *train* split only).

        Strided, not the first `n_max`: step 0 of every trajectory is the zero
        initial condition, so a prefix fit sees `log1p(0) = 0` everywhere and
        reports std[`u`] = 6e-4 (§3.5).
        """
        self.normalizer = Normalizer.fit(self, n_max=n_max)
        return self.normalizer

    # ------------------------------------------------------------------ h5 io
    @property
    def fh(self) -> h5py.File:
        if self._fh is None:
            self._fh = h5py.File(self.path, "r")
        return self._fh

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None  # a live h5py handle is not picklable (Windows + workers)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fh = None

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _read_params(self) -> dict[str, np.ndarray]:
        """Name the `/params` columns from the file, not from our own tuple.

        Reading by position means a regenerated file with one extra column hands
        `theta` the `rho_b` values and nothing anywhere notices -- the balance
        module would just use a 1.5-porosity. The `columns` attr costs 60 bytes.
        """
        with h5py.File(self.path, "r") as fh:
            if "/params" not in fh:
                return {}
            arr = np.asarray(fh["/params"][:], dtype=np.float64)
            cols = fh["/params"].attrs.get("columns")
            if cols is None:
                names = list(PARAM_COLUMNS)
            else:
                names = [c.decode() if isinstance(c, (bytes, np.bytes_)) else str(c) for c in cols]
            if arr.shape[1] != len(names):
                raise ValueError(
                    f"{self.path}: /params has {arr.shape[1]} columns but {len(names)} names {names}"
                )
            return {n: arr[:, i] for i, n in enumerate(names)}

    def _scalars(self, tid: int) -> dict[str, np.ndarray]:
        out = {}
        attrs = self.fh[f"/{self.traj_names[tid]}"].attrs
        for key in ("theta", "rho_b", "smax", "k_lang", "lam", "dt"):
            if key in self.params:
                out[key] = np.asarray(self.params[key][tid], dtype=np.float64)
            elif key in attrs:
                out[key] = np.asarray(float(attrs[key]))
            elif key == "dt":
                out[key] = np.asarray(float(self.dt))
            else:
                # a silent 1.0 here is a physical parameter that is off by 2x
                raise KeyError(f"{self.path}: no source for scalar '{key}' (neither /params nor attrs)")
        return out

    def _fields(self, tid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = self.fh[f"/{self.traj_names[tid]}"]
        q = np.stack([np.asarray(g["qx"][:], np.float32), np.asarray(g["qy"][:], np.float32)], 0)
        D = np.asarray(g["D"][:], np.float32)
        return q, D, g["src"][:]

    # -------------------------------------------------------------- samples
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        tid, n = self.index[i]
        g = self.fh[f"/{self.traj_names[tid]}"]
        u = np.asarray(g["C"][n], np.float32)[None]
        src = np.asarray(g["src"][n], np.float32)[None]
        # [K+1, 1, N, N] over steps n .. n+K, i.e. row 0 *is* `u` (§2, §3.11).
        # Returning only n+1..n+K forced the trainer to re-prepend `u`, and two
        # places that both decide what row 0 means eventually disagree.
        target = np.asarray(g["C"][n: n + self.K + 1], np.float32)[:, None]
        q, D, _ = self._fields(tid)
        sc = self._scalars(tid)
        batch = {
            "u": torch.from_numpy(u).to(self.dtype),
            "q": torch.from_numpy(q).to(self.dtype),
            "D": torch.from_numpy(D).to(self.dtype),
            "s": torch.from_numpy(src).to(self.dtype),
            # the K-step training rollout advances the *source* too; reusing one
            # step's source K times would bias the rollout loss toward whatever
            # the source happened to be at the window start
            "src_seq": torch.from_numpy(
                np.asarray(g["src"][n: n + self.K], np.float32)
            ).to(self.dtype)[:, None],
            "theta": self._scalar_tensor(sc["theta"]),
            "rho_b": self._scalar_tensor(sc["rho_b"]),
            "smax": self._scalar_tensor(sc["smax"]),
            "k_lang": self._scalar_tensor(sc["k_lang"]),
            "lam": self._scalar_tensor(sc["lam"]),
            "dt": float(sc["dt"]),
            "traj_id": tid,
            "step": n,
        }
        return batch, torch.from_numpy(target).to(self.dtype)

    def _scalar_tensor(self, v) -> torch.Tensor:
        return torch.tensor([float(v)], dtype=self.dtype).view(1, 1, 1)

    # -------------------------------------------------------------- rollouts
    def get_window(self, traj_id: int, start: int = 0, steps: int = 40) -> tuple[dict, torch.Tensor]:
        """Rollout-ready batch: same dict plus `src_seq` and long-horizon `ref`.

        Used by `evaluate/rollout.py`; `ref` is [1, steps+1, 1, N, N].
        """
        name = self.traj_names[int(traj_id)]
        g = self.fh[f"/{name}"]
        T = g["C"].shape[0] - 1
        steps = min(int(steps), max(T - start, 1))
        # [B=1, C=1, N, N] and [B=1, T, C=1, N, N]: the same ranks the loader
        # produces after `collate`, so a single trajectory is a batch of one and
        # the evaluator never has to special-case anything
        u = np.asarray(g["C"][start], np.float32)[None, None]
        src_seq = np.asarray(g["src"][start: start + steps], np.float32)[None, :, None]
        ref = np.asarray(g["C"][start: start + steps + 1], np.float32)[None, :, None]
        q, D, _ = self._fields(int(traj_id))
        sc = self._scalars(int(traj_id))
        batch = {
            "u": torch.from_numpy(u).to(self.dtype),
            "q": torch.from_numpy(q[None]).to(self.dtype),
            "D": torch.from_numpy(D[None]).to(self.dtype),
            "s": torch.from_numpy(src_seq[:, 0]).to(self.dtype),
            "src_seq": torch.from_numpy(src_seq).to(self.dtype),
            "theta": self._scalar_tensor(sc["theta"]),
            "rho_b": self._scalar_tensor(sc["rho_b"]),
            "smax": self._scalar_tensor(sc["smax"]),
            "k_lang": self._scalar_tensor(sc["k_lang"]),
            "lam": self._scalar_tensor(sc["lam"]),
            "dt": float(sc["dt"]),
            "traj_id": int(traj_id),
            "steps": steps,
            "origin": tuple(float(v) for v in self._origin(name)),
        }
        target = torch.from_numpy(ref).to(self.dtype)
        return batch, target

    def _origin(self, name: str) -> tuple[float, float]:
        attrs = self.fh[f"/{name}"].attrs
        return float(attrs.get("x0", 0.5 * self.N)), float(attrs.get("y0", 0.5 * self.N))

    def split_ids_for(self, split: str) -> np.ndarray:
        with h5py.File(self.path, "r") as fh:
            return np.asarray(fh[f"/splits/{split}"][:], dtype=np.int64)


class ZeroShotDataset(ADRDataset):
    """Read-only view of an extrapolation file (ext / kappa / long)."""

    def __init__(self, path: str, stats_path: str | None = None, K: int = 1,
                 split: str = "val") -> None:
        super().__init__(path=path, split=split, K=K, stats_path=stats_path)
        if self.regime == "train":
            raise ValueError(f"{path} has regime='train'; use ADRDataset for training data")
        if self.split_ids.size == 0:
            raise ValueError(f"{path} split '{split}' is empty — check /splits")


def collate(batch: Iterable[tuple[dict, torch.Tensor]]):
    """Default collate cannot handle the python-float `dt` (and traj metadata)."""
    batch = list(batch)
    keys = [k for k in batch[0][0] if k not in ("dt", "traj_id", "step", "origin", "steps")]
    out = {}
    for k in keys:
        vals = [b[0][k] for b in batch]
        if all(torch.is_tensor(v) for v in vals):
            out[k] = torch.stack(vals, 0)
        else:
            out[k] = vals
    dts = {b[0]["dt"] for b in batch}
    if len(dts) != 1:
        raise ValueError(f"batch mixes dt values: {sorted(dts)}")
    out["dt"] = float(dts.pop())
    out["traj_id"] = [b[0].get("traj_id") for b in batch]
    targets = torch.stack([b[1] for b in batch], 0)
    return out, targets
