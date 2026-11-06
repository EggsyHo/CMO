"""Calibrate the mass audit: inject a known error, check it is reported as such.

`r = |M(t) - M(0) - integral(sources - lambda*M)| / |M(0)|` is a global mass
bookkeeping residual. A headline of `r = 6e-6` is only meaningful if the estimator
responds proportionally to an error of that size, and only if it is *not* doing
something else (a self-fulfilling identity, a degenerate normalisation). Two checks,
both on an existing checkpoint -- no retraining:

  (a) rescale the field by `1 + eps` from the midpoint of the window onward, which
      adds exactly `eps * M` to the budget with no matching source term, and read
      `r`. Expected slope 1 in `eps`, no floor above the unperturbed residual.
  (b) destroy the field's shape while preserving its mass exactly (transpose each
      frame). `r` must stay at the unperturbed floor while `rel_l2` explodes -- that
      is a *limitation* stated in the paper, not a bug: the audit is a mass
      identity, and shape accuracy is `rel_l2`'s job.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from src.evaluate.evaluator import Evaluator          # noqa: E402
from src.evaluate.rollout import audit_series, rollout  # noqa: E402
from src.utils.metrics import rel_l2                   # noqa: E402

CKPT = "results/cmo_s0/cmo_adr2d_0/best.pth"
N_TRAJ = 3
EPS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)

ev = Evaluator(CKPT, steps=200, device="cuda" if torch.cuda.is_available() else "cpu")
ds, split = ev._dataset("test")
ids = ds.split_ids_for(split)[:N_TRAJ]
print(f"{CKPT}  |  {N_TRAJ} trajectories  |  {ev.steps} steps")
print("\n(a) injected relative mass error eps -> reported r")
print("  eps        r_final mean   r/eps    (r/eps ~ 1 means the estimator is honest)")
base = {}
for tid in ids:
    batch, ref = ds.get_window(int(tid), 0, ev.steps)
    out = rollout(ev.model, batch, ev.steps, ev.dx, device=ev.device)
    base[int(tid)] = (batch, out, ref.numpy()[:, :, 0])

for eps in EPS:
    r = []
    for tid, (batch, out, _ref) in base.items():
        pred = out["pred"].clone()
        t0 = ev.steps // 2
        pred[:, t0:] *= (1.0 + eps)
        r.append(float(audit_series(pred, batch, ev.dx, budget="composite")["r_final"][0]))
    r = np.array(r)
    ratio = "  n/a " if eps == 0 else f"{r.mean() / eps:8.3f}"
    print(f"  {eps:<9.0e} {r.mean():.3e}   {ratio}")

print("\n(b) shape destroyed, mass preserved (transpose every frame)")
r, e = [], []
for tid, (batch, out, ref) in base.items():
    pred = out["pred"].transpose(3, 4).contiguous()      # same M, different field
    r.append(float(audit_series(pred, batch, ev.dx, budget="composite")["r_final"][0]))
    e.append(float(rel_l2(pred.numpy()[:, :, 0], ref)))
print(f"  transposed: r_final {np.mean(r):.3e}   rel_l2_final {np.mean(e):.4f}")
r0, e0 = [], []
for tid, (batch, out, ref) in base.items():
    r0.append(float(audit_series(out["pred"], batch, ev.dx, budget="composite")["r_final"][0]))
    e0.append(float(rel_l2(out["pred"].numpy()[:, :, 0], ref)))
print(f"  original  : r_final {np.mean(r0):.3e}   rel_l2_final {np.mean(e0):.4f}")

print("\n(c) aqueous-only budget on the same rollout (why it must not be quoted as a control)")
ra = [float(audit_series(out["pred"], batch, ev.dx, budget="aqueous")["r_final"][0])
      for batch, out, _ref in base.values()]
print(f"  r_aqueous {np.mean(ra):.3e}  vs  r_composite {np.mean(r0):.3e}")
print("  -> the sorbed phase is a genuine sink for the aqueous budget; this number is")
print("     a property of the physical partitioning, not of numerical drift.")
