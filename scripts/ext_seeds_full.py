"""Five seeds on the fine-timestep band over the **full** 80-case set.

The 16-case replication (`bound_dt_seeds.py`, `ext_seeds_baselines.py`) fixed the
seed problem but left the level on a strict subset of the band, and the 16- and
80-case orderings of the same two arms disagree -- so neither the level nor the
ranking could be quoted without naming a subset. This runs every cell on all 80
trajectories, which is the only way the two can share one case set.

Four cells, one process each (the model is 265k parameters; they run concurrently
without contending):
  af_corr   cmo_af, bound x dt/dt_train
  af_raw    cmo_af, as trained          (needed for the paired improvement at n=80)
  fluxnet   FluxNet, as trained
  cmo       CMO, as trained

Resumable per seed: each seed's four cells are independent, and a dropped process
rewrites only what is missing.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from src.evaluate.evaluator import Evaluator        # noqa: E402
from src.evaluate.rollout import rollout            # noqa: E402
from src.utils.metrics import rel_l2                # noqa: E402

N = 80
CELLS = {
    "af_corr": ("results/cmo_af_s{s}/cmo_adr2d_{s}_cmo_af/best.pth", 0.16),
    "af_raw": ("results/cmo_af_s{s}/cmo_adr2d_{s}_cmo_af/best.pth", 1.0),
    "fluxnet": ("results/{d}fluxnet_adr2d_{s}/best.pth", 1.0),
    "cmo": ("results/cmo_s{s}/cmo_adr2d_{s}/best.pth", 1.0),
}

ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True, choices=list(CELLS))
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()

pat, scale = CELLS[a.cell]
# FluxNet's seed 0 predates the per-seed run dirs and lives under s11/.
d = "s11/" if (a.cell == "fluxnet" and a.seed == 0) else (
    f"fluxnet_s{a.seed}/" if a.cell == "fluxnet" else "")
ckpt = pat.format(s=a.seed, d=d)
out = f"results/_ext80_{a.cell}_s{a.seed}.json"
if os.path.exists(out):
    print(f"[skip] {out}")
    sys.exit(0)

ev = Evaluator(ckpt, device="cuda")
if a.cell.startswith("af"):
    ev.model._bound_time_scale = scale
ds, split = ev._dataset("ext")
ids = ds.split_ids_for(split)[:N]
errs = []
for i, tid in enumerate(ids):
    batch, ref = ds.get_window(int(tid), 0, 1250)
    o = rollout(ev.model, batch, 1250, ev.dx, device=ev.device)
    p = o["pred"].numpy()[:, :, 0][:, -1]
    t = ref.numpy()[:, :, 0][:, -1]
    errs.append(float(rel_l2(p, t, axis=(-2, -1)).mean()))
    if (i + 1) % 20 == 0:
        print(f"  {a.cell} s{a.seed} {i+1}/{N} running mean {np.mean(errs):.4f}", flush=True)

json.dump({"cell": a.cell, "seed": a.seed, "ckpt": ckpt, "scale": scale, "errs": errs},
          open(out, "w"), indent=1)
print(f"[done] {a.cell} s{a.seed} n={len(errs)} mean {np.mean(errs):.4f}", flush=True)
