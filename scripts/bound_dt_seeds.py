"""Is the dt repair a property of the bound, or of one checkpoint?

Every `ext` number in the study comes from seed 0, including the 76% collapse and the
0.4402 repair. The rest of the paper quotes five-seed means, so a headline claim built
on one checkpoint is exactly the asymmetry a referee looks for. This repeats the
1.00-vs-0.16 comparison on all five seeds of form B and reports the per-seed paired
difference, so the claim can be stated as a five-seed mean with a seed-level sd.

16 trajectories per cell rather than 80: the effect is 30%+ and paired, so the test
does not need the whole set, and 5 seeds x 2 scales x 16 is already ~40 min. The point
here is *consistency across initialisations*, not precision on any one of them -- the
80-trajectory number stays the published one.

Writes one entry per (arm, seed) and resumes, so a dropped process costs one seed.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from src.evaluate.evaluator import Evaluator        # noqa: E402
from src.evaluate.rollout import rollout            # noqa: E402
from src.utils.metrics import rel_l2                # noqa: E402

OUT = "results/_bound_dt_seeds.json"
N = 16
SCALES = [1.0, 0.16]
ARMS = {"cmo_af": "results/cmo_af_s{s}/cmo_adr2d_{s}_cmo_af/best.pth",
        "cmo_af_nosep": "results/cmo_af_nosep_s{s}/cmo_adr2d_{s}_cmo_af_nosep/best.pth"}

res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def run(ckpt, scale, ids, ds, ev):
    ev.model._bound_time_scale = float(scale)
    errs = []
    for tid in ids:
        batch, ref = ds.get_window(int(tid), 0, 1250)
        o = rollout(ev.model, batch, 1250, ev.dx, device=ev.device)
        p = o["pred"].numpy()[:, :, 0][:, -1]       # rollout: [B, T+1, 1, N, N]
        t = ref.numpy()[:, :, 0][:, -1]
        errs.append(float(rel_l2(p, t, axis=(-2, -1)).mean()))
    return errs


for arm, pat in ARMS.items():
    for s in range(5):
        key = f"{arm}_s{s}"
        if key in res:
            continue
        ev = Evaluator(pat.format(s=s), device="cuda")
        ds, split = ev._dataset("ext")
        ids = ds.split_ids_for(split)[:N]
        cell = {}
        for sc in SCALES:
            cell[f"{sc:.2f}"] = run(pat.format(s=s), sc, ids, ds, ev)
        a = np.asarray(cell["1.00"])
        b = np.asarray(cell["0.16"])
        res[key] = cell
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"{key:20s} 1.00 {a.mean():.4f}  0.16 {b.mean():.4f}  "
              f"{(b.mean()/a.mean()-1)*100:+6.1f}%  wins {int((b < a).sum())}/{N}", flush=True)

print("\nper-arm summary (seed means)")
for arm in ARMS:
    keep = [res[f"{arm}_s{s}"] for s in range(5) if f"{arm}_s{s}" in res]
    if len(keep) < 5:
        continue
    base = np.array([np.mean(c["1.00"]) for c in keep])
    corr = np.array([np.mean(c["0.16"]) for c in keep])
    rel = (corr / base - 1) * 100
    print(f" {arm:15s} baseline {base.mean():.4f} -> corrected {corr.mean():.4f}  "
          f"{rel.mean():+.1f}% +- {rel.std(ddof=1):.1f}% (seed sd, 5/5 seeds: "
          f"{int((rel < 0).sum())})")
