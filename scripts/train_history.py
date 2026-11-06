"""Why is `nfv`'s seed-to-seed sd 20x smaller than everyone else's?

Reads training histories (no retraining). Two competing explanations, distinguishable
from the curves: five seeds *converging to the same point* (optimiser artefact) versus
five seeds *plateauing at the same level* (capacity ceiling). The sign of the last-3
epoch slope separates them better than the endpoint alone.

Neither turned out to hold -- the cause is a seed-independent transport lag measured in
`scripts/centroid_drift.py` (docs/dev_log.md, 2026-11-03). This script is kept because
it is the cheapest way to rule the two out again on a new arm.
"""
import json

import numpy as np

ARMS = {
    "nfv": ["results/s11/nfv_adr2d_0"] + [f"results/nfv_s{s}/nfv_adr2d_{s}" for s in range(1, 5)],
    "fluxnet": ["results/s11/fluxnet_adr2d_0"] + [f"results/fluxnet_s{s}/fluxnet_adr2d_{s}" for s in range(1, 5)],
    "cmo": [f"results/cmo_s{s}/cmo_adr2d_{s}" for s in range(5)],
    "cmo_af": [f"results/cmo_af_s{s}/cmo_adr2d_{s}_cmo_af" for s in range(5)],
}

for arm, ps in ARMS.items():
    print(f"\n=== {arm} ===")
    print(" seed ep stop            rl_roll@e0  rl_roll@last  last-3 slope  val-tr gap%")
    ends = []
    for i, p in enumerate(ps):
        try:
            with open(f"{p}/train_info.json") as fh:
                d = json.load(fh)
        except FileNotFoundError:
            print(f"  {i}  missing")
            continue
        h = d["history"]
        rl = np.array([r["rel_l2_rollout"] for r in h], float)
        tl = np.array([r["train_loss"] for r in h], float)
        vl = np.array([r["val_loss"] for r in h], float)
        k = min(3, len(rl) - 1)
        slope = (rl[-1] - rl[-1 - k]) / k
        ends.append(rl[-1])
        gap = (vl[-1] - tl[-1]) / abs(tl[-1]) * 100
        print(f"  {i}   {len(h):2d} {str(d.get('stop_reason'))[:14]:14s} "
              f"{rl[0]:.5f}     {rl[-1]:.5f}      {slope:+.2e}    {gap:+.1f}")
    e = np.array(ends, float)
    print(f"  final rel_l2_rollout: mean {e.mean():.5f}  sd {e.std(ddof=1):.5f}  "
          f"spread {(e.max()-e.min())/e.mean()*100:.2f}%  epochs {len(e)} seeds")
