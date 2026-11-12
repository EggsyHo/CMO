"""Audit the released ADR-2D file against the claims in the manuscript.

Referees for a numerical-methods journal will re-derive a few things from the
data file itself.  This script does exactly those checks and prints a verdict,
so that the numbers quoted in the paper are reproducible from the archive:

  (1) material heterogeneity -- spatial spread of |q| and of the dispersion
      tensor, which documents the stationary-medium statement of the paper;
  (2) discrete divergence of the stored velocity (the property the mass audit
      relies on);
  (3) composition of the injected mass, i.e. sum(src)*dx^2 versus the physical
      rate, which exposes any per-cell renormalisation of the source;
  (4) the composite-mass audit on the stored trajectories.

    python scripts/audit_dataset.py [--file data/adr2d/adr2d.h5] [--n 20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "data"))
import gen_adr2d as g  # noqa: E402


def _fd_div(qx, qy, dx):
    return ((qx - np.roll(qx, 1, axis=-1)) / dx
            + (qy - np.roll(qy, 1, axis=-2)) / dx)


def audit_traj(fh, name, dt):
    grp = fh[name]
    qx, qy = grp["qx"][:], grp["qy"][:]
    D, C, src = grp["D"][:], grp["C"][:], grp["src"][:]
    at = grp.attrs
    dx = float(fh["meta"].attrs["dx"])
    qm = np.hypot(qx, qy)
    div = _fd_div(qx, qy, dx)
    scale = qm.mean() / dx

    th, rb = float(at["theta"]), float(at["rho_b"])
    sm, kl, lam = float(at["smax"]), float(at["k_lang"]), float(at["lam"])
    m = th * C + rb * g.langmuir(C, sm, kl)
    M = m.sum(axis=(-2, -1)) * dx ** 2
    inj = src.sum(axis=(-2, -1)) * dx ** 2
    expect = M[0] + np.cumsum(dt * (inj - lam * M[:-1]))
    return {
        "variance_ratio_q": float(qm.std() / qm.mean()),
        "variance_ratio_D": float(np.abs(D.std(axis=(1, 2)) / D.mean(axis=(1, 2))).max()),
        "max_div_q_over_scale": float(np.abs(div).max() / scale),
        "source_sum_per_step_first": float(inj[0]),
        "source_sum_per_step_last": float(inj[-1]),
        # dx**2 factor: injected mass is (Q * (1-e) ) * dx**2 in the generator
        "source_sum_over_dx2_last": float(inj[-1] / dx ** 2),
        "mass_audit_resid": float(np.abs(M[1:] - expect).max()
                                  / max(np.abs(M).max(), 1e-12)),
        "peak_C": float(C.max()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ADR-2D dataset audit")
    ap.add_argument("--file", default="data/adr2d/adr2d.h5")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/verification/data_audit.json")
    args = ap.parse_args(argv)

    with h5py.File(args.file, "r") as fh:
        names = [k for k in fh.keys() if k.startswith("traj_")][: args.n]
        dx = float(fh["meta"].attrs["dx"])
        dt = float(fh["meta"].attrs["dt"])
        ver = fh["meta"].attrs.get("generator_version", "?")
        rows = {n: audit_traj(fh, n, dt) for n in names}

    keys = ("variance_ratio_q", "variance_ratio_D", "max_div_q_over_scale",
            "source_sum_per_step_last", "mass_audit_resid", "peak_C")
    print(f"file {args.file}  dx={dx}  generator {ver}  trajectories {len(rows)}")
    print(f"{'check':28s} {'min':>11s} {'median':>11s} {'max':>11s}")
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in rows.values()])
        summary[k] = {"min": float(v.min()), "median": float(np.median(v)),
                      "max": float(v.max())}
        print(f"{k:28s} {v.min():11.3e} {np.median(v):11.3e} {v.max():11.3e}")
    inj = np.array([r["source_sum_over_dx2_last"] for r in rows.values()])
    print(f"\nsum(src)*dx^2 at the last step: {inj.min():.4f} .. {inj.max():.4f}")
    print("  (the generator divides the well profile by its discrete sum, not by")
    print("   its integral, so this quantity carries a factor dx^2; at dx = 1 it")
    print("   coincides with the physical rate Q(1-exp(-t/t_ramp)))")

    verdict = {
        "velocity_uniform": summary["variance_ratio_q"]["max"] < 1e-4,
        "D_uniform": summary["variance_ratio_D"]["max"] < 1e-4,
        "divergence_free": summary["max_div_q_over_scale"]["max"] < 1e-8,
        "mass_audit_ok": summary["mass_audit_resid"]["max"] < 1e-6,
    }
    print(f"\nverdict: {verdict}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"dx": dx, "generator_version": str(ver), "summary": summary,
                   "verdict": verdict}, fh, indent=2)
    print(f"wrote {args.out}")
    if not all(verdict.values()):
        print("FAIL: the archive violates a stated property of the benchmark")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
