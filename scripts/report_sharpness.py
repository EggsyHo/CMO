"""Report the sharpness estimator across arms and seeds (n=5, seed as the unit).

Reads what `scripts/recompute_sharpness.py` wrote. `ratio = w_model / w_ref` is
measured **at the reference's front**, so <1 means thinner than the reference at the
same place and same step -- not "keeps the front sharper", which is a claim about
accuracy and needs its own evidence (see docs/dev_log.md, 2026-11-03).
"""
import json

import numpy as np

T95 = 2.776
ARMS = ["fluxnet", "cmo", "nfv", "fno"]


def load(arm, field="sharpness_grad"):
    out = []
    for s in range(5):
        with open(f"results/_sharp/{arm}_s{s}/metrics_test.json") as fh:
            out.append(json.load(fh)[field])
    return out


print("sharpness_grad: w in cells, ratio = w_model / w_ref at the same step")
print(" arm       w_med   ratio_med  sd_ratio  ratio_final  frac")
vals = {}
for a in ARMS:
    r = load(a)
    rm = np.array([x["ratio_median"] for x in r])
    rf = np.array([x["ratio_final"] for x in r])
    wm = np.array([x["w_median"] for x in r])
    fr = np.array([x["frac_median"] for x in r])
    vals[a] = (rm, rf)
    print(f" {a:8s} {wm.mean():6.3f}  {rm.mean():8.3f}  {rm.std(ddof=1):8.3f}  "
          f"{rf.mean():10.3f}  {fr.mean():.3f}")

print("\npaired ratio differences (negative = thinner than the other arm):")
for a, b in (("cmo", "fluxnet"), ("nfv", "fluxnet"), ("fno", "fluxnet"), ("cmo", "nfv")):
    d = vals[a][0] - vals[b][0]
    se = d.std(ddof=1) / np.sqrt(d.size)
    lo, hi = d.mean() - T95 * se, d.mean() + T95 * se
    print(f"  {a:7s} - {b:7s}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"wins {int((d < 0).sum())}/5  {'SIG' if (lo > 0 or hi < 0) else 'ns'}")

print("\nper-seed ratio_final (does sharpness track accuracy?)")
for a in ARMS:
    print(f"  {a:8s} " + " ".join(f"{x:.3f}" for x in vals[a][1]))
print("\ncorrelation across seeds of ratio_median with rel_l2_final:")
for a in ARMS:
    err = np.array([json.load(open(f"results/_sharp/{a}_s{s}/metrics_test.json"))
                    ["rel_l2"]["final"] for s in range(5)])
    r = np.corrcoef(vals[a][0], err)[0, 1]
    print(f"  {a:8s} r(ratio, err) = {r:+.3f}   (n=5 critical |r|=0.878)")
