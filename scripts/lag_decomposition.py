"""Split the 200-step error into displacement and shape, by optimal translation.

Motivation (docs/dev_log.md, 2026-11-03): every arm under-transports -- the mass
centroid trails the reference by 1.16-1.40 cells -- and the arms' error *growth*
rates are identical, so `rel_l2` at 200 steps is dominated by a common lag. That
raises the obvious reviewer question: is the 3.1% gain of the detail branch a shape
improvement, or does it just move a bit more mass?

Answer by re-measuring after removing the displacement: for each trajectory, search
integer shifts on the periodic grid within +-MAXSHIFT cells, take the best, and
report

    E_total   = rel_l2(pred, ref)                        as published (k=200)
    E_shape   = min_shift rel_l2(roll(pred, shift), ref)
    E_disp    = E_total - E_shape                         (>= 0 by construction)
    shift*    = the argmin, to compare against the measured centroid drift

Periodic `roll` is legitimate: the solver is periodic and the metrics already sample
with `mode="grid-wrap"`. The shift is chosen per trajectory, never per arm -- a
per-arm shift would let an arm buy error reduction with one free parameter, which is
exactly the thing being tested. `rel_l2` is imported from `src.utils.metrics` so the
`E_total` column reproduces the published number instead of defining a new metric.
"""
import json
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, ".")
from src.utils.metrics import rel_l2   # noqa: E402

MAXSHIFT = 4
T95 = 2.776
def _runs(arm):
    """Five seed directories, seed 0 of the baselines living in the step-11 tree.

    All seven arms must cover the same five seeds: the main table quotes five-seed
    means, so a decomposition built on seed 0 alone would print `CMO − FluxNet`
    as 5.7% while the table says 6.4%, and the two numbers would sit in one paper.
    """
    if arm in ("fluxnet", "nfv", "fno"):
        return [f"results/s11/{arm}_adr2d_0"] + [f"results/{arm}_s{s}/{arm}_adr2d_{s}"
                                                 for s in range(1, 5)]
    suf = "" if arm == "cmo" else f"_{arm}"
    return [f"results/{arm}_s{s}/cmo_adr2d_{s}{suf}" for s in range(5)]


ARMS = ["cmo", "cmo_nosep", "cmo_af", "cmo_af_nosep", "fluxnet", "nfv", "fno"]


def final_fields(path):
    """rollout npz is [B, 1, T, 1, N, N]; return the last step as [B, N, N]."""
    d = np.load(f"{path}/rollout_test.npz")
    return d["pred"][:, 0, -1, 0], d["ref"][:, 0, -1, 0]


def decompose(pred, ref):
    tot = rel_l2(pred, ref, axis=(-2, -1))
    shape, shifts = [], []
    for b in range(pred.shape[0]):
        cand, arg = np.inf, (0, 0)
        for di in range(-MAXSHIFT, MAXSHIFT + 1):
            for dj in range(-MAXSHIFT, MAXSHIFT + 1):
                e = rel_l2(np.roll(pred[b], (di, dj), axis=(0, 1)), ref[b])
                if e < cand:
                    cand, arg = float(e), (di, dj)
        shape.append(cand)
        shifts.append(arg)
    return tot, np.array(shape), np.array(shifts)


print(f"optimal translation within +-{MAXSHIFT} cells (periodic), per trajectory, k=200")
print(f"{'arm':15s} {'E_total':>8s} {'E_shape':>8s} {'E_disp':>8s} {'gain%':>6s} "
      f"{'|shift*|':>8s}  top shifts")
res = {}
for arm in ARMS:
    tots, shps, shs = [], [], []
    missing = False
    for p in _runs(arm):
        try:
            pred, ref = final_fields(p)
        except FileNotFoundError:
            missing = True
            break
        a, b, c = decompose(pred, ref)
        tots.append(a)
        shps.append(b)
        shs.append(c)
    if missing or not tots:
        print(f" {arm:15s} missing")
        continue
    tot, shp, sh = np.concatenate(tots), np.concatenate(shps), np.concatenate(shs)
    res[arm] = (tot, shp)
    mag = np.linalg.norm(sh, axis=1)
    top = Counter(map(tuple, sh)).most_common(2)
    print(f" {arm:15s} {tot.mean():8.4f} {shp.mean():8.4f} {(tot - shp).mean():+8.4f} "
          f"{(tot - shp).mean() / tot.mean() * 100:6.1f} {mag.mean():8.2f}  "
          + ", ".join(f"{k}x{v}" for k, v in top))


def paired(a, b, idx):
    d = res[a][idx] - res[b][idx]
    se = d.std(ddof=1) / np.sqrt(d.size)
    return (d.mean(), d.mean() - T95 * se, d.mean() + T95 * se,
            int((d < 0).sum()), d.size)


print("\npaired over trajectories, E_total (negative = first arm better):")
for a, b in (("cmo", "cmo_nosep"), ("cmo_af", "cmo_af_nosep"),
             ("cmo", "fluxnet"), ("nfv", "fluxnet"), ("fno", "fluxnet")):
    if a not in res or b not in res:
        continue
    for idx, tag in ((0, "E_total"), (1, "E_shape")):
        m, lo, hi, w, n = paired(a, b, idx)
        rel = m / res[b][idx].mean() * 100
        sig = "SIG" if (lo > 0 or hi < 0) else "ns "
        print(f"  {a:13s} - {b:13s} {tag:8s}: {m:+.5f} [{lo:+.5f}, {hi:+.5f}]  "
              f"{rel:+.2f}%  wins {w}/{n}  {sig}")

print("\nthe 3.1% separation claim, re-tested after removing displacement:")
for hi_arm, lo_arm, name in (("cmo", "cmo_nosep", "form A"),
                             ("cmo_af", "cmo_af_nosep", "form B")):
    if hi_arm not in res or lo_arm not in res:
        continue
    for idx, tag in ((0, "E_total"), (1, "E_shape")):
        m, lo, hi, w, n = paired(hi_arm, lo_arm, idx)
        print(f"  {name} {tag:8s}: {m / res[lo_arm][idx].mean() * 100:+.2f}%  "
              f"wins {w}/{n}  {'SIG' if (lo > 0 or hi < 0) else 'ns'}")

with open("results/_lag_decomposition.json", "w", encoding="utf-8") as fh:
    json.dump({k: {"E_total": v[0].tolist(), "E_shape": v[1].tolist()}
               for k, v in res.items()}, fh, indent=2)
print("\nwrote results/_lag_decomposition.json")
