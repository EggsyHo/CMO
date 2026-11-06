"""Is the flux bound's out-of-band failure a one-off, or does it grow with exposure?

The long-horizon zero-shot showed `cmo_af` (absolute flux bound) 37% worse than `cmo`
(detail branch, relative flux) at T=1250, while inside the band the two are within 1%.
A single point cannot separate "this band is hard" from "the bound is wrong here", so
this reads the *stored* 1250-step series at several truncation lengths and reports how
the penalty accumulates. `rollout_ext.npz` keeps the whole trajectory, so every
intermediate horizon is free -- no extra rollouts.

The prediction being tested is specific: a bound `|T| <= 0.5 * m_upwind` that is
calibrated to the training amplitudes does not fail by a constant offset, it fails by
being *exceeded* wherever the local flux is larger than the calibration. Mass then
accumulates in place instead of moving, which is an error that grows with the number
of steps the bound is exceeded. So the penalty should be ~flat early and increase
monotonically with T. A flat curve at long horizon would mean the bound is not the
cause and the finding would have to be withdrawn.

Reported as the paired difference (bounded minus unbounded, same architecture, same
seed, same trajectories), which removes everything except the flux parametrisation.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from src.utils.metrics import rel_l2   # noqa: E402

PAIRS = [("cmo_af", "cmo", "form B bound ON vs OFF"),
         ("cmo_af_nosep", "cmo_nosep", "bound ON vs OFF, no detail branch")]
ARMS = ["cmo", "cmo_nosep", "cmo_af", "cmo_af_nosep", "fluxnet"]
HORIZONS = [50, 100, 200, 400, 800, 1250]


def series(path, key="pred"):
    """Mean-over-trajectories rel_l2 at every stored step, [T]."""
    d = np.load(path)
    p, r = d["pred"][:, 0, :, 0], d["ref"][:, 0, :, 0]   # [B, T, N, N]
    per = rel_l2(p, r, axis=(-2, -1))                    # [B, T]
    return per.mean(axis=0), per


band = {}
for a in ARMS:
    band[a] = series(f"results/_zshot/ext_{a}/rollout_ext.npz")

print("ext band (Pe -> 1), mean rel_l2 vs rollout length, n=80 trajectories")
print(f"{'T':>6s}" + "".join(f"{a:>11s}" for a in ARMS))
for T in HORIZONS:
    if T >= band["cmo"][0].size:
        continue
    print(f"{T:6d}" + "".join(f"{band[a][0][T]:11.4f}" for a in ARMS))

print("\npenalty from the flux bound alone (paired, same trajectories)")
print(f"{'T':>6s}" + "".join(f"{p[2][:18]:>20s}" for p in PAIRS))
for T in HORIZONS:
    if T >= band["cmo"][0].size:
        continue
    row = f"{T:6d}"
    for hi, lo, _ in PAIRS:
        d = band[hi][1][:, T] - band[lo][1][:, T]
        rel = d.mean() / band[lo][1][:, T].mean() * 100
        se = d.std(ddof=1) / np.sqrt(d.size)
        t95 = 2.776 if d.size < 300 else 1.96
        sig = "" if abs(d.mean()) > t95 * se else " ns"
        row += f"  {d.mean():+.4f} ({rel:+5.1f}%){sig:<4s}"
    print(row)

print("\nthe same two arms inside the training band, for contrast")
for hi, lo, tag in PAIRS:
    s_hi = series(f"results/{'cmo_af_s0/cmo_adr2d_0_cmo_af' if hi == 'cmo_af' else 'cmo_af_nosep_s0/cmo_adr2d_0_cmo_af_nosep'}/rollout_test.npz")
    s_lo = series(f"results/{'cmo_s0/cmo_adr2d_0' if lo == 'cmo' else 'cmo_nosep_s0/cmo_adr2d_0_cmo_nosep'}/rollout_test.npz")
    for T in (50, 100, 200):
        d = s_hi[1][:, T] - s_lo[1][:, T]
        rel = d.mean() / s_lo[1][:, T].mean() * 100
        print(f"  {tag:32s} T={T:4d}: {d.mean():+.4f} ({rel:+5.2f}%)")
