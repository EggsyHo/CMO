"""Is form B's out-of-band collapse caused by the bound's missing dt?

Structure, not opinion (see the `flux_form == "absolute"` branch in cmo.py):
    relative:  du = -dt * div(F)
    absolute:  du = -div(tanh(F) * 0.5 * m_upwind)        <-- no dt anywhere
so the absolute branch advances by a fixed fraction of the upwind mass *per call*.
The bound therefore silently encodes a timestep: it is calibrated to whatever dt the
training data used. `adr2d.h5` integrates at dt=0.100, `adr2d_ext.h5` at dt=0.016 --
the same physical time (200 x 0.1 = 1250 x 0.016 = 20), 6.25x more calls. If the bound
is the cause of the 76% degradation, the same weights should move ~6.25x too much mass
per unit time in that band, and scaling the bound by dt/dt_train = 0.16 should undo it.

That is a single-knob intervention on the existing forward pass -- no retraining --
because the multiplier is a constant on a path that is already there (`_bound_time_scale`,
default 1.0, so every published number is untouched). It is a *test* rather than a knob
search only if the control runs too: the same 0.16 applied inside the training band,
where dt is the one the bound was calibrated with, must make things **worse**. A
correction that helps in both bands would be a free parameter, not a correction.

Equally important negative control: `adr2d_kappa.h5` also runs at dt=0.100, which is why
form B is fine there (40/40 at kappa=20). That single fact already rules out "absolute
bounds don't transfer" as the general explanation -- kappa changes by 20x and the bound
survives -- leaving dt as the axis that differs.

Prediction, stated before the run:
  ext   (dt=0.016): scale 0.16 -> error falls from ~0.673 toward the dt-blind arms (~0.49)
  band  (dt=0.100): scale 0.16 -> error rises well above 0.381
If scale 0.16 does not help at ext, the dt hypothesis is wrong and the finding is not
written.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from src.evaluate.evaluator import Evaluator        # noqa: E402
from src.evaluate.rollout import rollout            # noqa: E402
from src.utils.metrics import rel_l2                # noqa: E402

CKPT = "results/cmo_af_s0/cmo_adr2d_0_cmo_af/best.pth"
EXT_STORED = "results/_zshot/ext_cmo_af/rollout_ext.npz"
DT_TRAIN, DT_EXT = 0.100, 0.016


def centroid(pred, ref):
    """Signed mass-weighted centroid displacement along +x, in cells."""
    def c(a):
        w = a.sum(-2)                                    # [B, N] mass per column
        x = np.arange(w.shape[-1], dtype=np.float64)
        return (w * x).sum(-1) / np.maximum(w.sum(-1), 1e-12)
    return float((c(pred) - c(ref)).mean())


def run_band(ev, mode, steps, scale, max_traj=None):
    """rel_l2 and centroid at the final frame, over one split."""
    ev.model._bound_time_scale = float(scale)          # default 1.0 = published behaviour
    ds, split = ev._dataset(mode)
    ids = ds.split_ids_for(split)
    if max_traj:
        ids = ids[:max_traj]
    errs, cents = [], []
    for tid in ids:
        batch, ref = ds.get_window(int(tid), 0, steps)
        out = rollout(ev.model, batch, steps, ev.dx, device=ev.device)
        p = out["pred"].numpy()[:, :, 0]               # [B, T+1, N, N]
        r = ref.numpy()[:, :, 0]
        errs.append(float(rel_l2(p[:, -1], r[:, -1], axis=(-2, -1)).mean()))
        cents.append(centroid(p[:, -1], r[:, -1]))
    return float(np.mean(errs)), float(np.mean(cents)), len(ids)


ev = Evaluator(CKPT, device="cuda")

# ---- stored reference for the fine-dt band, so the intervention is compared with the
# ---- number that is already in the paper rather than with a re-run of itself
d = np.load(EXT_STORED)
p, r = d["pred"][:, 0, -1, 0], d["ref"][:, 0, -1, 0]
ext_ref = (float(rel_l2(p, r, axis=(-2, -1)).mean()), centroid(p, r))
print(f"ext, scale 1.00 (stored, published): rel_l2 {ext_ref[0]:.4f}  centroid {ext_ref[1]:+.3f} cells")

print(f"\n{'band':22s} {'dt':>6s} {'scale':>6s} {'rel_l2':>8s} {'centroid':>10s} {'n':>4s}")
for mode, steps, tag, dt in (("test", 200, "in-band", DT_TRAIN),
                             ("ext", 1250, "ext", DT_EXT)):
    for sc in (1.0, 0.16):
        if mode == "ext" and sc == 1.0:
            continue                                   # stored above
        e, c, n = run_band(ev, mode, steps, sc)
        print(f" {tag:22s} {dt:6.3f} {sc:6.2f} {e:8.4f} {c:+10.3f} {n:4d}")

json.dump({"ext_stored": ext_ref}, open("results/_bound_dt_intervention.json", "w"), indent=1)
