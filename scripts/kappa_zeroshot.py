"""Zero-shot check on the dispersivity sweep (kappa x1 and x20), never run before.

Every claim in the paper so far is measured on one band of the training regime. The
cheapest external test the design already built for is `adr2d_kappa.h5`: the same
generator, same solver, same grid, with the dispersion coefficient at the training
value and at 20x. Nothing is retrained; each focal checkpoint is rolled out 200 steps
on that file and scored with the published metrics.

Two questions, and they are different:

  (1) Does conservation survive a 20x change in the operator's coefficients? `r` is
      integrated from the analytic source every step, so this is a test of the
      composite variable, not of the architecture.
  (2) Does the detail branch's shape gain survive out of distribution? This is the
      only positive claim in the paper, so a gain that exists at kappa x1 and
      vanishes at x20 would be a much weaker result than the within-regime ablation.

Reported per family rather than pooled: a mean over x1 and x20 would average away
exactly the thing worth knowing, since x20 is where a truncated spectral head is
expected to fail first (it cannot build the high-wavenumber amplitude that stronger
diffusion needs).

n=1 seed (seed 0), so no seed-level CI: the intervals here are trajectory-level and
exist only to say whether the sign is consistent across cases, not to estimate a
seed-level effect size.
"""
import json
import sys

import h5py
import numpy as np

sys.path.insert(0, ".")
from src.utils.metrics import rel_l2   # noqa: E402

ARMS = {
    "cmo": "results/_zshot/kappa_cmo",
    "cmo_nosep": "results/_zshot/kappa_cmo_nosep",
    "cmo_af": "results/_zshot/kappa_cmo_af",
    "cmo_af_nosep": "results/_zshot/kappa_cmo_af_nosep",
    "fluxnet": "results/_zshot/kappa_fluxnet",
    "nfv": "results/_zshot/kappa_nfv",
    "fno": "results/_zshot/kappa_fno",
}
T95 = 2.776


def kappa_of(h5="data/adr2d/adr2d_kappa.h5"):
    """Per-trajectory kappa, in file order.

    `/params` is an [n, 9] array whose columns are named by its `columns` attr —
    reading by position would silently take `Da` for `kappa` on any file that
    reorders them, so the names are read from the file (`adr_dataset._read_params`
    does the same for exactly this reason).
    """
    with h5py.File(h5, "r") as f:
        arr = np.asarray(f["params"][:], dtype=float)
        cols = [c.decode() if isinstance(c, bytes) else str(c)
                for c in f["params"].attrs["columns"]]
    if "kappa" not in cols:
        raise SystemExit(f"no kappa column in {cols}")
    return arr[:, cols.index("kappa")], "kappa"


def final_err(path):
    d = np.load(f"{path}/rollout_test.npz")
    return rel_l2(d["pred"][:, 0, -1, 0], d["ref"][:, 0, -1, 0], axis=(-2, -1))


kap, kname = kappa_of()
print(f"kappa per trajectory from `params/{kname}`: uniques {np.unique(kap)}")
groups = {f"kappa={v:g}": (kap == v) for v in np.unique(kap)}

err, cons = {}, {}
for arm, p in ARMS.items():
    err[arm] = final_err(p)
    c = json.load(open(f"{p}/metrics_test.json"))["conservation"]
    cons[arm] = (c["r_final_mean"], c["r_aqueous_final_mean"])
    if err[arm].size != kap.size:
        raise SystemExit(f"{arm}: {err[arm].size} trajectories vs {kap.size} kappa entries")

print("\nrel_l2 at k=200, zero-shot (n=1 seed, 80 trajectories)")
hdr = f"{'arm':15s} {'all':>8s}" + "".join(f"{g:>12s}" for g in groups)
print(hdr)
for arm in ARMS:
    row = f" {arm:15s} {err[arm].mean():8.4f}"
    for g, m in groups.items():
        row += f"{err[arm][m].mean():12.4f}"
    print(row)

print("\nconservation residual r_final / r_aqueous (should be seed- and kappa-independent)")
for arm, (r, ra) in cons.items():
    print(f" {arm:15s} {r:.3e}   aqueous {ra:8.3f}")


def paired(a, b, mask=None):
    d = err[a] - err[b]
    if mask is not None:
        d = d[mask]
    se = d.std(ddof=1) / np.sqrt(d.size)
    return d.mean(), d.mean() - T95 * se, d.mean() + T95 * se, int((d < 0).sum()), d.size


print("\nseparation effect, zero-shot (negative = with detail branch better)")
for hi, lo, name in (("cmo", "cmo_nosep", "form A"), ("cmo_af", "cmo_af_nosep", "form B")):
    for tag, mask in [("pooled", None)] + [(g, m) for g, m in groups.items()]:
        m, lo_ci, hi_ci, w, n = paired(hi, lo, mask)
        sig = "SIG" if (lo_ci > 0 or hi_ci < 0) else "ns "
        print(f"  {name} {tag:10s}: {m:+.5f} [{lo_ci:+.5f}, {hi_ci:+.5f}]  "
              f"{m / err[lo][mask].mean() * 100:+6.2f}%  wins {w}/{n}  {sig}")

print("\nhead structure, zero-shot (CMO vs the two conserving baselines)")
for b in ("fluxnet", "nfv"):
    for tag, mask in [("pooled", None)] + [(g, m) for g, m in groups.items()]:
        m, lo_ci, hi_ci, w, n = paired("cmo", b, mask)
        print(f"  cmo - {b:8s} {tag:10s}: {m:+.5f} [{lo_ci:+.5f}, {hi_ci:+.5f}]  "
              f"{m / err[b][mask].mean() * 100:+6.2f}%  wins {w}/{n}")
