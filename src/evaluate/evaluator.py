"""Evaluator: checkpoint -> numbers (implementation.md §3.15).

Everything here is computed from the checkpoint alone (`cfg` and the normalizer
are stored inside it), so a result can be reproduced from `best.pth` without the
config file that produced it. The two things that are *asserted* rather than
reported:

* the rebuilt model has the parameter count of the stored weights -- a silent
  config drift would otherwise be indistinguishable from a different model;
* whenever the diffusion-only width is available it is printed next to the
  measured width, and a width below it is labelled, not hidden.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..data.adr_dataset import ADRDataset
from ..data.transforms import Normalizer
from ..models import build_model
from ..utils import metrics
from ..utils.config import config_hash, freeze, get
from .rollout import audit_series, rollout

MODES = ("test", "ext", "negative_control")


def _param_count_from_state(state: dict, param_names) -> int:
    """Parameters actually present in the checkpoint (buffers excluded).

    Keyed on the model's own parameter names, not on a `weight|bias` suffix: the
    operator carries bare `nn.Parameter`s (`log_flux_scale`, gate `beta/tau`), and
    a suffix rule silently under-counts them by 30x, which is how the first
    version of this check "detected" a mismatch on a perfectly good checkpoint.
    """
    return int(sum(v.numel() for k, v in state.items() if k in param_names))


class Evaluator:
    def __init__(self, ckpt_path: str, data_path: str | None = None, steps: int = 40,
                 keep_time: bool = False, device: str = "cpu") -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.ckpt = ckpt
        self.cfg = freeze(ckpt["cfg"])
        self.device = torch.device(device)
        self.steps = int(steps)
        self.keep_time = keep_time
        self.dx = float(get(self.cfg, "data.dx", 1.0))
        self.block = int(get(self.cfg, "model.block", 2))

        stats = ckpt.get("normalizer")
        if stats and not os.path.isabs(stats):
            stats = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), stats)
        self.normalizer = Normalizer.load(stats) if stats and os.path.exists(stats) else None
        self.model = build_model(self.cfg, self.normalizer, verbose=False)
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        expected = _param_count_from_state(ckpt["model"],
                                           {n for n, _ in self.model.named_parameters()})
        if int(self.model.n_params()) != expected:
            raise RuntimeError(f"parameter count mismatch: model {self.model.n_params()} "
                               f"vs checkpoint {expected}")
        self.model.to(self.device).eval()
        self.run_id = str(ckpt.get("run_id") or os.path.basename(os.path.dirname(ckpt_path)))
        self.data_path = data_path or str(get(self.cfg, "data.path"))

    # ------------------------------------------------------------------ data
    def _dataset(self, mode: str) -> tuple[ADRDataset, str]:
        if mode == "test":
            return ADRDataset(self.data_path, "test", K=1), "test"
        if mode == "ext":
            path = str(get(self.cfg, "eval.ext_path", "")) or self._sibling("ext")
            return ADRDataset(path, "val", K=1), "val"
        path = str(get(self.cfg, "eval.neg_path", "")) or self._sibling("negative")
        return ADRDataset(path, "val", K=1), "val"

    def _sibling(self, kind: str) -> str:
        base = os.path.dirname(os.path.abspath(self.data_path))
        name = {"ext": "adr2d_ext.h5", "negative": "adr2d_negative.h5"}[kind]
        path = os.path.join(base, name)
        if not os.path.exists(path):
            raise FileNotFoundError(mode_path_msg(kind, path))
        return path

    # ------------------------------------------------------------------ main
    def run(self, mode: str = "test", out_dir: str | None = None,
            max_traj: int | None = None) -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
        ds, split = self._dataset(mode)
        traj_ids = ds.split_ids_for(split)[: max_traj or len(ds.split_ids_for(split))]
        if len(traj_ids) == 0:
            raise RuntimeError(f"split '{split}' is empty in {ds.path}")

        per = []
        store, store_ref = [], []
        for tid in traj_ids:
            batch, ref = ds.get_window(int(tid), 0, self.steps)
            out = rollout(self.model, batch, self.steps, self.dx, device=self.device)
            per.append(self._trajectory_metrics(out, ref, batch))
            store.append(out["pred"].numpy())
            store_ref.append(ref.numpy())

        agg = _aggregate(per)
        agg.update({"run_id": self.run_id, "mode": mode, "data": os.path.basename(ds.path),
                    "n_traj": int(len(per)), "steps": int(self.steps),
                    "model": str(get(self.cfg, "model.name", "cmo")),
                    "param_count": int(self.model.n_params()),
                    "config_hash": config_hash(self.cfg)})
        if out_dir:
            self._write(out_dir, mode, agg, np.stack(store), np.stack(store_ref), per)
        return agg

    def _trajectory_metrics(self, out: dict, ref: torch.Tensor, batch: dict) -> dict:
        pred = out["pred"].numpy()[:, :, 0]                    # [1, T, N, N]
        r = ref.numpy()[:, :, 0]
        origin = batch.get("origin")
        dt = float(batch["dt"])
        T = pred.shape[1]

        # per *step*, not per sample: `pred[0]` is [T,N,N], so the spatial axes are
        # (1,2). Passing the batch axes as well collapses the series to length 1.
        rel = metrics.rel_l2(pred[0], r[0], axis=(1, 2))
        rho = metrics.rho_series(pred[0], self.block)
        rho_ref = metrics.rho_series(r[0], self.block)
        tv = np.array([metrics.tv_ratio(pred[0, k], r[0, k]) for k in range(T)])
        width, valid, width_ref = [], [], []
        for k in range(T):
            w, nv = metrics.front_width(pred[0, k], origin)
            wr, _ = metrics.front_width(r[0, k], origin)
            width.append(w)
            width_ref.append(wr)
            valid.append(nv)
        # §5 sharpness, second estimator: `front_width` normalises per field by
        # min/max, which reads peak height as width (`w/w_ref` tracked the decay
        # curve, not the interface). This one is amplitude-free, and it is measured
        # *at the reference's front* for both fields (`interface_width_pair`), so a
        # displaced front or a spurious spike cannot silently change which edge is
        # being measured. `frac` is the coverage flag, stored so a suspicious ratio can
        # be traced to a truncated edge instead of being quoted blindly.
        iw, iw_ref, iw_frac = [], [], []
        for k in range(T):
            w, wr, f = metrics.interface_width_pair(pred[0, k], r[0, k])
            iw.append(w)
            iw_ref.append(wr)
            iw_frac.append(f)
        err = pred[0, -1] - r[0, -1]
        e_coarse, e_detail = metrics.band_split(err, self.block)
        audit = audit_series(out["pred"], batch, self.dx, budget="composite")
        audit_aq = audit_series(out["pred"], batch, self.dx, budget="aqueous")
        bt = metrics.breakthrough(
            pred[0], outlet_col=(int(origin[1]) if origin else None), dt=dt)
        return {
            "rel_l2_series": rel, "tv_series": tv, "rho_series": rho,
            "rho_ref_series": rho_ref, "width_series": np.asarray(width),
            "width_ref_series": np.asarray(width_ref), "valid_rays": valid,
            "iw_series": np.asarray(iw), "iw_ref_series": np.asarray(iw_ref),
            "iw_frac_series": np.asarray(iw_frac),
            "audit_r": audit["r"][0], "audit_r_aq": audit_aq["r"][0],
            "audit_degenerate": bool(audit["degenerate"][0]),
            "e_coarse": e_coarse, "e_detail": e_detail, "bt": bt, "dt": dt,
        }

    def _write(self, out_dir: str, mode: str, agg: dict, pred: np.ndarray,
               ref: np.ndarray, per: list[dict]) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"metrics_{mode}.json"), "w", encoding="utf-8") as fh:
            json.dump(agg, fh, indent=2, default=float)
        np.savez_compressed(os.path.join(out_dir, f"rollout_{mode}.npz"),
                            pred=pred.astype(np.float32), ref=ref.astype(np.float32))
        rows = ["t,rel_l2,tv_ratio,rho,rho_ref,width,width_ref,audit_r,audit_r_aqueous"]
        for k in range(self.steps + 1):
            rows.append(",".join([
                str(k),
                f"{_nanmean([p['rel_l2_series'][k] for p in per]):.6e}",
                f"{_nanmean([p['tv_series'][k] for p in per]):.6e}",
                f"{_nanmean([p['rho_series'][k] for p in per]):.6e}",
                f"{_nanmean([p['rho_ref_series'][k] for p in per]):.6e}",
                f"{_nanmean([p['width_series'][k] for p in per]):.6e}",
                f"{_nanmean([p['width_ref_series'][k] for p in per]):.6e}",
                f"{_nanmean([p['audit_r'][k] for p in per]):.6e}",
                f"{_nanmean([p['audit_r_aq'][k] for p in per]):.6e}",
            ]))
        with open(os.path.join(out_dir, f"audit_{mode}.csv"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")


def mode_path_msg(kind: str, path: str) -> str:
    return (f"{kind} data not found at {path}; generate it with "
            f"`python data/gen_adr2d.py --regime {kind}`")


def _nanmean(values) -> float:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _nanratio(width: np.ndarray, width_ref: np.ndarray) -> float:
    """Median over steps of the per-step width ratio.

    `front_width` returns NaN on a ray that does not cross both levels (a flat or
    single-cell front), so the ratio must be formed *per step first* and medianed
    second -- taking means of the two series before dividing lets one NaN step
    silently drop out of one series and not the other.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = width / np.where(np.abs(width_ref) > 1e-30, width_ref, np.nan)
    arr = ratio[np.isfinite(ratio)]
    return float(np.median(arr)) if arr.size else float("nan")


def _aggregate(per: list[dict]) -> dict:
    rel = np.stack([p["rel_l2_series"] for p in per])
    tv = np.stack([p["tv_series"] for p in per])
    width = np.stack([p["width_series"] for p in per])
    width_ref = np.stack([p["width_ref_series"] for p in per])
    iw = np.stack([p["iw_series"] for p in per])
    iw_ref = np.stack([p["iw_ref_series"] for p in per])
    frac = np.stack([p["iw_frac_series"] for p in per])
    audit = np.stack([p["audit_r"] for p in per])
    audit_aq = np.stack([p["audit_r_aq"] for p in per])
    return {
        "rel_l2": {"step1": _nanmean(rel[:, 1]),
                   "final": _nanmean(rel[:, -1]),
                   "final10": _nanmean(rel[:, -10:]),
                   "series": [_nanmean(rel[:, k]) for k in range(rel.shape[1])],
                   "growth_rate": metrics.error_growth_rate(
                       np.array([_nanmean(rel[:, k]) for k in range(rel.shape[1])]))},
        "conservation": {"r_final_mean": _nanmean(audit[:, -1]),
                         "r_final_max": float(np.nanmax(audit[:, -1])),
                         "r_aqueous_final_mean": _nanmean(audit_aq[:, -1]),
                         "budget": "composite",
                         "n_degenerate_windows": int(sum(p["audit_degenerate"] for p in per))},
        "width": {"final": _nanmean(width[:, -1]), "ref_final": _nanmean(width_ref[:, -1]),
                  "ratio": float(_nanmean(width[:, -1]) / max(_nanmean(width_ref[:, -1]), 1e-30))},
        # §5 sharpness: a per-step ratio, not just the last step. A model that is
        # sharp at k=1 and blurred by k=8 has the same `width.ratio` as one that is
        # uniformly 10% wide, and only the median separates them.
        "sharpness": {"w_ratio_median": _nanratio(width, width_ref),
                      "w_ratio_last": float(_nanmean(width[:, -1])
                                            / max(_nanmean(width_ref[:, -1]), 1e-30)),
                      # the ray count behind `w_ratio_last`, not the trajectory
                      # minimum: at k=0 the plume is a point source with no 10-90%
                      # front, so a min over steps would always read 0 and make a
                      # perfectly measured width look unverified.
                      "valid_rays": int(min(p["valid_rays"][-1] for p in per)),
                      "valid_rays_min": int(min(min(p["valid_rays"]) for p in per))},
        # §5, the estimator that actually measures width (see metrics.interface_width).
        # `iw_ratio` > 1 means thicker than the reference at the same step; the series
        # is kept because a model can be sharp at k=3 and smeared by k=100, and the
        # median alone hides which.
        "sharpness_grad": {
            "w_median": _nanmean(iw.reshape(-1)),
            "w_ref_median": _nanmean(iw_ref.reshape(-1)),
            "ratio_median": _nanratio(iw, iw_ref),
            "ratio_final": float(_nanmean(iw[:, -1]) / max(_nanmean(iw_ref[:, -1]), 1e-30)),
            "series": [_nanmean(iw[:, k] / iw_ref[:, k]) for k in range(iw.shape[1])],
            "frac_median": _nanmean(frac.reshape(-1)),
        },
        "rho": {"step1": _nanmean(np.stack([p["rho_series"][1] for p in per])),
                "final": _nanmean(np.stack([p["rho_series"][-1] for p in per])),
                "ref_final": _nanmean(np.stack([p["rho_ref_series"][-1] for p in per]))},
        "spectral": {"rho_init": _nanmean(np.stack([p["rho_series"][1] for p in per])),
                     "rho_final": _nanmean(np.stack([p["rho_series"][-1] for p in per]))},
        "tv_ratio": {"final": _nanmean(tv[:, -1]), "mean": _nanmean(tv.reshape(-1))},
        "tv": {"ratio_final": _nanmean(tv[:, -1])},
        "band": {"e_coarse": _nanmean([p["e_coarse"] for p in per]),
                 "e_detail": _nanmean([p["e_detail"] for p in per])},
        "breakthrough": {k: _nanmean([p["bt"][k] for p in per])
                         for k in per[0]["bt"] if isinstance(per[0]["bt"][k], (int, float))},
    }
