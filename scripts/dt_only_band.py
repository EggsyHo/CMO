"""A dt-only band: the single-variable test the `ext` band could not give.

`ext` moves dt, Pe, Da and kappa together, so "the bound encodes the timestep" has
so far rested on exclusion plus an in-band sign flip. Here both files are drawn from
the training regime with the same 80 parameter vectors and the same seed, and differ
in nothing but the integration step:

  adr2d_dtcoarse.h5   200 steps x dt 0.100  = t 20
  adr2d_dtfine.h5    1250 steps x dt 0.016  = t 20

Parameter sampling depends only on (regime, n, seed), so the two files are paired
case by case: any change in an arm's error is attributable to `dt`, not to a shifted
Pe/Da/kappa distribution, and not to a different case set.

Also reports the reference-vs-reference difference at the final frame, which bounds
how much of any change is simply the finer integration being a slightly different
target rather than the model misbehaving.

Usage: python scripts/dt_only_band.py --cell af_corr --seeds 0,1,2,3,4
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from src.data.adr_dataset import ADRDataset      # noqa: E402
from src.evaluate.evaluator import Evaluator     # noqa: E402
from src.evaluate.rollout import rollout         # noqa: E402
from src.utils.metrics import rel_l2             # noqa: E402

BANDS = {"coarse": ("data/adr2d/adr2d_dtcoarse.h5", 200),
         "mid": ("data/adr2d/adr2d_dtmid.h5", 400),
         "fine": ("data/adr2d/adr2d_dtfine.h5", 1250)}
CKPT = {
    "af_corr": "results/cmo_af_s{s}/cmo_adr2d_{s}_cmo_af/best.pth",
    "af_raw": "results/cmo_af_s{s}/cmo_adr2d_{s}_cmo_af/best.pth",
    "af_nosep_corr": "results/cmo_af_nosep_s{s}/cmo_adr2d_{s}_cmo_af_nosep/best.pth",
    "af_nosep_raw": "results/cmo_af_nosep_s{s}/cmo_adr2d_{s}_cmo_af_nosep/best.pth",
    "cmo": "results/cmo_s{s}/cmo_adr2d_{s}/best.pth",
    "fluxnet": "results/{d}fluxnet_adr2d_{s}/best.pth",
    # trained on three files that share every parameter and differ only in `dt`
    # (0.100 / 0.050 / 0.020) -- the retraining arm the paper used to lack
    "mix_af": "results/cmo_af_mix_s{s}/cmo_adr2d_{s}_cmo_af_dtmix/best.pth",
    "mix_cmo": "results/cmo_mix_s{s}/cmo_adr2d_{s}_cmo_dtmix/best.pth",
}
CORR = {"af_corr", "af_nosep_corr"}


def ckpt_of(cell, seed):
    d = "s11/" if (cell == "fluxnet" and seed == 0) else (
        f"fluxnet_s{seed}/" if cell == "fluxnet" else "")
    return CKPT[cell].format(s=seed, d=d)


def ids_of(ds, part):
    """`train` is the 56 cases already scored; `holdout` is the other 24.

    Splitting the band this way lets the published 56 stay comparable while the
    remaining cases are added instead of re-run — 56 + 24 = the whole file.
    """
    if part == "train":
        return ds.split_ids_for("train")
    return np.concatenate([ds.split_ids_for("val"), ds.split_ids_for("test")])


def run(cell, seed, part="train", bands=("coarse", "fine")):
    suffix = "" if part == "train" else f"_{part}"
    out = f"results/_dtonly_{cell}_s{seed}{suffix}.json"
    if os.path.exists(out):
        print(f"[skip] {out}", flush=True)
        return
    ev = Evaluator(ckpt_of(cell, seed), device="cuda")
    if cell in CORR:
        ev.model._bound_time_scale = 0.16

    res = {"cell": cell, "seed": seed, "part": part, "ckpt": ckpt_of(cell, seed)}
    for band in bands:
        path, steps = BANDS[band]
        ds = ADRDataset(path, "train", K=1)
        ids = ids_of(ds, part)
        errs, refs = [], []
        for i, tid in enumerate(ids):
            batch, ref = ds.get_window(int(tid), 0, steps)
            o = rollout(ev.model, batch, steps, ev.dx, device=ev.device)
            p = o["pred"].numpy()[:, :, 0][:, -1]
            t = ref.numpy()[:, :, 0][:, -1]
            errs.append(float(rel_l2(p, t, axis=(-2, -1)).mean()))
            refs.append(t)
            if (i + 1) % 20 == 0:
                print(f"  {cell} s{seed} {band} {i+1}/{len(ids)} "
                      f"mean {np.mean(errs):.4f}", flush=True)
        res[band] = errs
        res[f"{band}_ref"] = np.stack(refs)
        print(f"[{band}] {cell} s{seed} {part} {np.mean(errs):.4f}", flush=True)

    # how much of the difference is the target itself moving
    rc = np.stack(res.pop("coarse_ref"))
    rf = np.stack(res.pop("fine_ref"))
    res["ref_shift"] = float(rel_l2(rf, rc, axis=(-2, -1)).mean())
    res.pop("mid_ref", None)
    json.dump(res, open(out, "w"), indent=1)
    summary = "  ".join(f"{b} {np.mean(res[b]):.4f}" for b in bands)
    print(f"[done] {cell} s{seed} {summary} ref-shift {res['ref_shift']:.4f}",
          flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True, choices=list(CKPT))
ap.add_argument("--seeds", default="0", help="comma separated, e.g. 0,1,2,3,4")
ap.add_argument("--part", default="train", choices=["train", "holdout", "both"])
ap.add_argument("--bands", default="coarse,fine",
                help="comma separated subset of coarse,mid,fine; the retrained "
                     "arms add `mid`, which lies inside their training set")
a = ap.parse_args()
parts = ["train", "holdout"] if a.part == "both" else [a.part]
bands = tuple(x.strip() for x in a.bands.split(",") if x.strip())
for s in [int(x) for x in a.seeds.split(",")]:
    for p in parts:
        run(a.cell, s, p, bands)
