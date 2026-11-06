"""Inventory / sanity-check the generated ADR-2D HDF5 files.

Used by `scripts/gen_data.sh` after each file is written, so the pipeline fails
loudly if a file was opened in the wrong mode, if a split went missing, or if a
label file silently lost trajectories. Needs `h5py` only -- it runs before torch
is involved.

    python data/check_data.py                                  # inventory only
    python data/check_data.py --expect adr2d.h5=400 --expect adr2d_ext.h5=80

The contract gate (keys, shapes, target alignment, index completeness, parameter
supports, normalizer round trip) lives in `tests/check_data.py`; this file only
counts what is on disk.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py

DEFAULT_DIR = "data/adr2d"


def inspect(path: str) -> dict:
    with h5py.File(path, "r") as f:
        meta = dict(f["/meta"].attrs)
        trajs = sorted(k for k in f if k.startswith("traj_"))
        splits = {s: len(f[f"/splits/{s}"][:]) for s in ("train", "val", "test")}
        qc = {}
        if trajs:
            for key in ("qc_mass_resid", "qc_cfl_diff"):
                vals = [float(f[t].attrs[key]) for t in trajs if key in f[t].attrs]
                if vals:
                    qc[key] = max(vals)
        return {"path": path, "meta": meta, "n_traj": len(trajs), "splits": splits,
                "qc": qc, "has_ref128": "/ref128" in f,
                "C_shape": tuple(f[f"/{trajs[0]}/C"].shape) if trajs else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--expect", action="append", default=[],
                    help="file_basename=count, repeatable; mismatch exits non-zero")
    ap.add_argument("--only", action="append", default=[],
                    help="print only these basenames (repeatable)")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.dir, "*.h5")))
    if args.only:
        files = [f for f in files if os.path.basename(f) in set(args.only)]
    if not files:
        print(f"no .h5 in {args.dir}")
        return 1

    seen = {}
    for path in files:
        info = inspect(path)
        seen[os.path.basename(path)] = info["n_traj"]
        m = info["meta"]
        size = os.path.getsize(path) / 2**20
        print(f"{os.path.basename(path):18s} {size:7.1f} MB  n={m.get('n_traj'):>3} "
              f"T={m.get('T'):>5} dt={m.get('dt')} regime={m.get('regime')} "
              f"traj={info['n_traj']} C={info['C_shape']} splits={info['splits']}")
        qc = "  ".join(f"{k}={v:.2e}" for k, v in info["qc"].items())
        print(f"    blocks={m.get('regime_blocks')}  ref128={info['has_ref128']}"
              + (f"  {qc}" if qc else ""))

    bad = []
    for spec in args.expect:
        name, _, want = spec.partition("=")
        got = seen.get(name)
        if got != int(want):
            bad.append(f"{name}: expected {want} trajectories, found {got}")
    for msg in bad:
        print("MISMATCH " + msg)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
