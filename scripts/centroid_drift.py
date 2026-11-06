"""Mass-centroid displacement: systematic transport bias vs random error.

Rationale (docs/dev_log.md, 2026-11-03): `nfv`'s seed-to-seed sd of `0.0003` on
`rel_l2` is anomalous next to `fluxnet`'s `0.0061`. If that is a seed-independent
*transport* bias -- the arm moves less mass, as the zero-source probe suggested --
then its plume centroid lags the reference by the same amount on every seed. A random
error would not be consistent, and a capacity wall would show up as a flat error
curve instead (see scripts/train_history.py).

The signed version (along the reference's own flow direction) is what distinguishes
"lags" from "leads"; `|dcentroid|` alone cannot. Both are printed.
"""
import numpy as np

ARMS = {
    "nfv": ["results/s11/nfv_adr2d_0"] + [f"results/nfv_s{s}/nfv_adr2d_{s}" for s in range(1, 5)],
    "fluxnet": ["results/s11/fluxnet_adr2d_0"] + [f"results/fluxnet_s{s}/fluxnet_adr2d_{s}" for s in range(1, 5)],
    "cmo": [f"results/cmo_s{s}/cmo_adr2d_{s}" for s in range(5)],
    "fno": ["results/s11/fno_adr2d_0"] + [f"results/fno_s{s}/fno_adr2d_{s}" for s in range(1, 5)],
}
N = 64
yy, xx = np.mgrid[0:N, 0:N]


def cent(f):
    m = f.sum()
    if m <= 1e-12:
        return np.array([np.nan, np.nan]), 0.0
    return np.array([(yy * f).sum() / m, (xx * f).sum() / m]), m


print(" arm      k    |dcentroid| pred-ref (cells)          mass ratio")
for arm, ps in ARMS.items():
    rows = []
    for p in ps:
        d = np.load(f"{p}/rollout_test.npz")
        pred, ref = d["pred"][:, 0, :, 0], d["ref"][:, 0, :, 0]
        disp, mr = {}, {}
        for k in (50, 100, 200):
            ds, rs = [], []
            for i in range(ref.shape[0]):
                cp, mp = cent(pred[i, k])
                cr, mrm = cent(ref[i, k])
                ds.append(np.linalg.norm(cp - cr))
                rs.append(mp / max(mrm, 1e-12))
            disp[k], mr[k] = np.array(ds), np.array(rs)
        rows.append((p, disp, mr))
    for k in (50, 100, 200):
        dm = np.array([np.nanmean(r[1][k]) for r in rows])
        mm = np.array([np.nanmean(r[2][k]) for r in rows])
        print(f" {arm:8s} {k:4d}   mean {dm.mean():5.2f} sd {dm.std(ddof=1):4.2f}  "
              f"per-seed " + " ".join(f"{x:.1f}" for x in dm) + f"   {mm.mean():.3f}")

print("\nsigned offset along the reference's flow direction (negative = under-transport)")
for arm, ps in ARMS.items():
    for k in (100, 200):
        sg = []
        for p in ps:
            d = np.load(f"{p}/rollout_test.npz")
            pred, ref = d["pred"][:, 0, :, 0], d["ref"][:, 0, :, 0]
            per = []
            for i in range(ref.shape[0]):
                cr, _ = cent(ref[i, k])
                c0, _ = cent(ref[i, 2])
                u = cr - c0
                u = u / np.linalg.norm(u)
                cp, _ = cent(pred[i, k])
                per.append(float((cp - cr) @ u))
            sg.append(np.mean(per))
        sg = np.array(sg)
        print(f" {arm:8s} k={k:3d}  mean {sg.mean():+.3f}  sd {sg.std(ddof=1):.3f}  "
              f"per-seed " + " ".join(f"{x:+.2f}" for x in sg))
