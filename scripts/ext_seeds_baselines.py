"""Five seeds of the two unbounded baselines on the fine-timestep band.

`bound_dt_seeds.py` showed that form B's failure there is a seed-scatter problem
(sd 0.148 -> 0.015 after the dt fix) rather than a uniform bias. The remaining
question is the one a referee asks next: where does the corrected arm *rank*? That
comparison exists only for seed 0, because the `ext` zero-shot was run for one
checkpoint per arm. This fills the two baselines that carry the paper's main
contrasts -- CMO and FluxNet -- on the same five seeds.

Same 16 trajectory ids and the same band as `bound_dt_seeds.py` on purpose: the two
json files are only comparable cell-by-cell if the ids match, and a different subset
would silently turn a ranking into a subset artefact. One scale per arm (neither arm
has a bound to rescale). Resumable per (arm, seed).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from src.evaluate.evaluator import Evaluator        # noqa: E402
from src.evaluate.rollout import rollout            # noqa: E402
from src.utils.metrics import rel_l2                # noqa: E402

OUT = "results/_ext_seeds_baselines.json"
N = 16
# seed 0 of the baselines predates the per-seed run dirs and lives under s11/
CKPT = {
    "cmo": {0: "results/cmo_s0/cmo_adr2d_0/best.pth",
            **{s: f"results/cmo_s{s}/cmo_adr2d_{s}/best.pth" for s in range(1, 5)}},
    "fluxnet": {0: "results/s11/fluxnet_adr2d_0/best.pth",
                **{s: f"results/fluxnet_s{s}/fluxnet_adr2d_{s}/best.pth" for s in range(1, 5)}},
}

res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for arm, per_seed in CKPT.items():
    for s, ckpt in per_seed.items():
        key = f"{arm}_s{s}"
        if key in res:
            continue
        if not os.path.exists(ckpt):
            print(f"[miss] {key}: {ckpt}", flush=True)
            continue
        ev = Evaluator(ckpt, device="cuda")
        ds, split = ev._dataset("ext")
        ids = ds.split_ids_for(split)[:N]
        errs = []
        for tid in ids:
            batch, ref = ds.get_window(int(tid), 0, 1250)
            o = rollout(ev.model, batch, 1250, ev.dx, device=ev.device)
            p = o["pred"].numpy()[:, :, 0][:, -1]
            t = ref.numpy()[:, :, 0][:, -1]
            errs.append(float(rel_l2(p, t, axis=(-2, -1)).mean()))
        res[key] = {"1.00": errs}
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"{key:14s} {np.mean(errs):.4f}", flush=True)

print("\nband level, five seeds (same 16 ids as bound_dt_seeds.py)")
for arm in CKPT:
    v = [np.mean(res[f"{arm}_s{s}"]["1.00"]) for s in range(5) if f"{arm}_s{s}" in res]
    if len(v) == 5:
        v = np.array(v)
        print(f" {arm:10s} {v.mean():.4f} +- {v.std(ddof=1):.4f}   per seed {np.round(v,4).tolist()}")
b = json.load(open("results/_bound_dt_seeds.json"))
for key in ("cmo_af", "cmo_af_nosep"):
    if f"{key}_s4" in b:
        corr = np.array([np.mean(b[f"{key}_s{s}"]["0.16"]) for s in range(5)])
        print(f" {key:10s} dt-corrected {corr.mean():.4f} +- {corr.std(ddof=1):.4f}")
