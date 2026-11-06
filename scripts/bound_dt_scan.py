"""Where does the bound's scaling optimum sit? At dt/dt_train, or just "smaller"?

`bound_dt_intervention.py` showed the sign flip the dt-coupling predicts: scaling the
absolute bound by dt/dt_train = 0.16 makes form B *worse* inside the training band
(+12.3%) and much better on the fine-dt band (0.6725 -> 0.4402, -34.6%). One point
cannot distinguish two explanations:

  (a) the bound silently encodes a timestep, so the correct scale is dt/dt_train;
  (b) `Pe -> 1` needs less transport in general, so any reduction helps and 0.16 is
      simply the first value tried.

Under (a) the error is minimised near 0.16 and rises on both sides. Under (b) it keeps
falling as the scale goes to zero. The scan straddles the predicted value
(0.5 / 0.16 / 0.05) and decides between them; if the minimum is at 0.05 the finding is
"over-transport", not "dt-coupling", and it gets written that way.

40 trajectories per scale (paired against the stored baseline on the same 40), which is
half the cost of the 80-trajectory run and still 40 paired differences.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from src.evaluate.evaluator import Evaluator        # noqa: E402
from src.evaluate.rollout import rollout            # noqa: E402
from src.utils.metrics import rel_l2                # noqa: E402

CKPT = "results/cmo_af_s0/cmo_adr2d_0_cmo_af/best.pth"
N = 40
SCALES = [0.5, 0.16, 0.05]

d = np.load("results/_zshot/ext_cmo_af/rollout_ext.npz")
base = rel_l2(d["pred"][:, 0, -1, 0][:N], d["ref"][:, 0, -1, 0][:N], axis=(-2, -1))
print(f"scale 1.00 (stored, n={N}): {base.mean():.4f}")

ev = Evaluator(CKPT, device="cuda")
ds, split = ev._dataset("ext")
ids = ds.split_ids_for(split)[:N]
out = {"1.00": {"mean": float(base.mean()), "per": base.tolist()}}

print(f"\n{'scale':>6s} {'rel_l2':>8s} {'vs 1.00':>9s} {'paired wins':>12s}")
for sc in SCALES:
    ev.model._bound_time_scale = float(sc)
    errs = []
    for tid in ids:
        batch, ref = ds.get_window(int(tid), 0, 1250)
        o = rollout(ev.model, batch, 1250, ev.dx, device=ev.device)
        # `rollout` returns [B, T+1, 1, N, N] (unlike the stored npz, which is
        # [B, 1, T, N, N]); squeeze the channel *then* take the last frame. Indexing
        # both with `[:, 0, -1, 0]` silently took frame 0 of the reference -- the empty
        # initial field -- whose relative error is 0/0 and therefore NaN.
        p = o["pred"].numpy()[:, :, 0][:, -1]
        t = ref.numpy()[:, :, 0][:, -1]
        errs.append(float(rel_l2(p, t, axis=(-2, -1)).mean()))
    e = np.asarray(errs)
    wins = int((e < base).sum())
    print(f" {sc:6.2f} {e.mean():8.4f} {(e.mean()/base.mean()-1)*100:+8.1f}% {wins:9d}/{N}")
    out[f"{sc:.2f}"] = {"mean": float(e.mean()), "per": errs}

json.dump(out, open("results/_bound_dt_scan.json", "w"), indent=1)
print("\ndone -> results/_bound_dt_scan.json")
