"""Control numbers of the four archive bands, recomputed from the stored parameters.

Table 1 of the paper quotes the 2nd and 98th percentiles of Pe_Delta, Da, R0 and
kappa for each band. This script reads the parameter table that every archive
carries and re-derives those intervals, so the table is a computed object rather
than a transcription.

    Pe_Delta = Delta x / alpha_T        (Delta x = 1 m)
    Da       = lambda L theta / |q|
    R0       = 1 + rho_b s_max k_Lang / theta
    kappa    = alpha_L / alpha_T

The archive stores the realised values of Pe, Da and kappa per trajectory, so the
only quantity that has to be assembled here is R0.

Usage: python scripts/control_numbers.py [--out results/verification/control_numbers.json]
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

DATA = "data/adr2d"
BANDS = {
    "training": ("adr2d.h5", None),
    "extrapolation": ("adr2d_ext.h5", None),
    "kappa=1": ("adr2d_kappa.h5", 1.0),
    "kappa=20": ("adr2d_kappa.h5", 20.0),
}
PCT = (2.0, 98.0)


def params_of(name):
    with h5py.File(os.path.join(DATA, name), "r") as fh:
        p = fh["params"][:]
        cols = [c.decode() if isinstance(c, bytes) else str(c)
                for c in fh["params"].attrs.get("columns", [])]
    if cols is None:      # the generator writes a fixed column order
        cols = ["Pe", "Da", "kappa", "theta", "rho_b", "smax", "k_lang", "lam", "dt"]
    return {c: p[:, i] for i, c in enumerate(cols)}


def band_stats(name, kappa=None):
    p = params_of(name)
    if kappa is not None:
        keep = np.isclose(p["kappa"], kappa)
        p = {k: v[keep] for k, v in p.items()}
    r0 = 1.0 + p["rho_b"] * p["smax"] * p["k_lang"] / p["theta"]
    out = {"n": int(p["Pe"].size)}
    for key, value in (("Pe_delta", p["Pe"]), ("Da", p["Da"]), ("R0", r0),
                       ("kappa", p["kappa"])):
        lo, hi = np.percentile(value, PCT)
        out[key] = [float(lo), float(hi)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/verification/control_numbers.json")
    args = ap.parse_args()

    res = {band: band_stats(name, k) for band, (name, k) in BANDS.items()}
    for band, d in res.items():
        print(f"{band:<14} n={d['n']:<4} "
              f"Pe {d['Pe_delta'][0]:.1f}-{d['Pe_delta'][1]:.1f}  "
              f"Da {d['Da'][0]:.2f}-{d['Da'][1]:.2f}  "
              f"R0 {d['R0'][0]:.2f}-{d['R0'][1]:.2f}  "
              f"kappa {d['kappa'][0]:.1f}-{d['kappa'][1]:.1f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"percentiles": list(PCT), "bands": res}, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
