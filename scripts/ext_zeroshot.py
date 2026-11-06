"""Summarise the long-horizon zero-shot (`ext`, T=1250) once the driver finishes.

Reads `results/_zshot/ext_<arm>/`, one seed (seed 0, n=1), and reports what the paper
needs at this band: the level, the paired contrasts that carry the claims, and the
audit residual, which is **not** comparable to the 200-step value and is shown only so
that its band-dependence is on the record.

The residual deserves a word: every arm returns `r ≈ 4.86e-5` to three digits at
T=1250, against `6.99e-6` at T=200. Arms that differ by 45% in `rel_l2` cannot share a
model-specific mass error to three digits, so this growth is the audit's own
time-discretisation accumulation over 6.25x more steps (dt=0.016, 1250 quadrature
increments), not a property of any architecture. Quoting `4.9e-5` as "conservation
degrades at long horizon" would therefore be a misreading — the honest statement is
that `r` has a step-count floor, which is why §3.15 reports it per band.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from src.utils.metrics import rel_l2   # noqa: E402

ARMS = ["cmo", "cmo_nosep", "cmo_af", "cmo_af_nosep", "fluxnet", "nfv", "fno"]
IN_BAND = {"cmo": 0.3845, "cmo_nosep": 0.3974, "cmo_af": 0.3810,
           "cmo_af_nosep": 0.3923, "fluxnet": 0.3615, "nfv": 0.4338, "fno": 0.6191}
T95 = 2.776

err, rfin = {}, {}
for a in ARMS:
    p = f"results/_zshot/ext_{a}"
    try:
        d = np.load(f"{p}/rollout_ext.npz")
        err[a] = rel_l2(d["pred"][:, 0, -1, 0], d["ref"][:, 0, -1, 0], axis=(-2, -1))
        rfin[a] = json.load(open(f"{p}/metrics_ext.json"))["conservation"]["r_final_mean"]
    except FileNotFoundError:
        print(f" (pending) {a}")

print(f"{'arm':15s} {'ext T=1250':>11s} {'in-band':>8s} {'change':>8s} {'r_final':>10s}")
for a in err:
    ch = err[a].mean() / IN_BAND[a] - 1
    print(f" {a:15s} {err[a].mean():11.4f} {IN_BAND[a]:8.4f} {ch*100:+7.1f}% {rfin[a]:10.3e}")


def paired(a, b):
    d = err[a] - err[b]
    se = d.std(ddof=1) / np.sqrt(d.size)
    return (d.mean(), d.mean() - T95 * se, d.mean() + T95 * se,
            int((d < 0).sum()), d.size, d.mean() / err[b].mean() * 100)


print("\npaired over the 80 trajectories (n=1 seed; sign consistency, not a seed CI)")
for a, b, tag in (("cmo", "cmo_nosep", "separation A"),
                  ("cmo_af", "cmo_af_nosep", "separation B"),
                  ("cmo", "fluxnet", "CMO vs FluxNet"),
                  ("cmo", "nfv", "CMO vs NFV"),
                  ("nfv", "fluxnet", "NFV vs FluxNet"),
                  ("fno", "fluxnet", "FNO vs FluxNet")):
    if a not in err or b not in err:
        continue
    m, lo, hi, w, n, rel = paired(a, b)
    print(f"  {tag:16s} {m:+.5f} [{lo:+.5f}, {hi:+.5f}]  {rel:+6.2f}%  "
          f"wins {w}/{n}  {'SIG' if (lo > 0 or hi < 0) else 'ns'}")
