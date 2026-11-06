"""Shape-level confirmation: does the dt-normalised bound stop spreading the plume sideways?

The error numbers in `bound_dt_scan.py` say the correction helps; they do not say *how*.
At Pe -> 1 the reference plume is barely advected and spreads little across the flow,
and form B as trained reports a transverse (y) spread of 1.577x the reference while its
along-flow (x) spread is 0.885x -- i.e. it is not lagging or leading, it is bleeding mass
sideways. That is what "6.25x too much movement per unit time" should look like on a
nearly static field, and it is a prediction the correction can fail.

Same 40 trajectories as the scan, stored fields for scale 1.00, one re-roll for 0.16.
If the transverse ratio does not return toward 1 the dt account is incomplete and the
mechanism sentence in the paper gets rewritten around over-dispersion instead.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from src.evaluate.evaluator import Evaluator        # noqa: E402
from src.evaluate.rollout import rollout            # noqa: E402

CKPT = "results/cmo_af_s0/cmo_adr2d_0_cmo_af/best.pth"
N = 40


def spread(a):
    """Centroid and per-axis standard deviation of the mass, [B, N, N] -> (x, sx, sy)."""
    x = np.arange(a.shape[-1], dtype=np.float64)
    m = a.sum((-2, -1)) + 1e-30
    cx = (a.sum(-2) * x).sum(-1) / m
    cy = (a.sum(-1) * x).sum(-1) / m
    sx = np.sqrt((a.sum(-2) * (x[None, :] - cx[:, None]) ** 2).sum(-1) / m)
    sy = np.sqrt((a.sum(-1) * (x[None, :] - cy[:, None]) ** 2).sum(-1) / m)
    return cx, sx, sy


d = np.load("results/_zshot/ext_cmo_af/rollout_ext.npz")
ref = d["ref"][:, 0, -1, 0][:N]
_, sxr, syr = spread(ref)

rows = {}
for tag, pred in (("1.00 (stored)", d["pred"][:, 0, -1, 0][:N]),):
    _, sx, sy = spread(pred)
    rows[tag] = (np.mean(sx / sxr), np.mean(sy / syr))

ev = Evaluator(CKPT, device="cuda")
ev.model._bound_time_scale = 0.16
ds, split = ev._dataset("ext")
ids = ds.split_ids_for(split)[:N]
preds = []
for tid in ids:
    batch, _ = ds.get_window(int(tid), 0, 1250)
    o = rollout(ev.model, batch, 1250, ev.dx, device=ev.device)
    preds.append(o["pred"].numpy()[:, :, 0][:, -1])
_, sx, sy = spread(np.concatenate(preds))
rows["0.16 (dt-scaled)"] = (np.mean(sx / sxr), np.mean(sy / syr))

print("plume spread relative to the reference, 40 trajectories at dt=0.016")
print(f"{'bound scale':18s} {'x (along flow)':>15s} {'y (across)':>12s}")
for k, (a, b) in rows.items():
    print(f" {k:18s} {a:15.3f} {b:12.3f}")
