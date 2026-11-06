"""Train at 64^2, evaluate at 128^2: does the operator carry over without retraining?

The claim this checks is narrow and worth stating before the code: `Pi` and the
flux update are *grid-defined*, so a checkpoint trained on 64^2 keeps working when
handed a 128^2 field, with the coarse grid becoming 64x64 of 2x2 blocks. Nothing
in the model is re-fitted, no parameter depends on `N`, and the fine grid only
enters through `dx`. Whether that is enough is an empirical question -- the
relative flux form multiplies by `dt` and divides by `dx`, so halving `dx`
doubles the transported distance per step at the same physical `dt`, and the
detail branch's CFL factor doubles with it and saturates its 0.25 floor in the
fastest cells. Both effects are expected to hurt; the number that matters is
how much.

Protocol, and the two choices in it:

  * the references are 60 *freshly drawn* trajectories from the same `train`
    regime, integrated at 128^2 -- not the 60 test trajectories re-integrated.
    Regenerating the stored trajectories at a finer grid needs the generator's
    RNG to be replayed block by block, which costs a full 400-trajectory 128^2
    run for the same statistical content. Fresh cases from the same parameter
    bands measure the same transfer, and they are labelled as fresh everywhere.
  * the solver timestep is halved (0.1 -> 0.05) because the diffusive CFL scales
    as `dt/dx^2` and would otherwise grow 4x; every second frame is kept, so the
    stored frame spacing stays 0.1 s and the model sees the same physical dt it
    was trained with. Without this the comparison would be dt-transfer, not
    resolution-transfer, and this paper already has a dt experiment.

Usage:
  python scripts/resolution_transfer.py --generate
  python scripts/resolution_transfer.py --evaluate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, ".")

OUT_H5 = "data/adr2d/adr2d_128test.h5"
RES_DIR = "results/_zshot"
ARMS = {"cmo": "cmo_s0", "cmo_nosep": "cmo_nosep_s0",
        "cmo_af": "cmo_af_s0", "cmo_af_nosep": "cmo_af_nosep_s0"}


def _ckpt(run_dir: str) -> str:
    import glob
    hits = sorted(glob.glob(os.path.join("results", run_dir, "*", "best.pth")))
    if not hits:
        raise FileNotFoundError(f"no best.pth under results/{run_dir}")
    return hits[0]


def generate(n: int, N: int, T_ref: int, dt_ref: float, stride: int,
             seed: int, chunk: int = 10, out: str = OUT_H5) -> str:
    """Integrate `n` fresh 128^2 trajectories and write an ADRDataset-shaped file."""
    sys.path.insert(0, "data")
    import gen_adr2d as G                                   # noqa: E402
    from src.data.adr_dataset import PARAM_COLUMNS          # noqa: E402

    rng = np.random.default_rng(seed)
    prm = G.sample_parameters_multi(rng, ["train"], [n])
    dx = G.DOMAIN / N
    T = T_ref // stride

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with h5py.File(out, "w") as fh:
        meta = fh.create_group("meta")
        meta.attrs.update({"N": N, "T": T, "dt": dt_ref * stride, "dx": dx,
                           "regime": "train", "seed": seed, "n_traj": n,
                           "generator_version": "1.0-res-transfer"})
        # `dt` is not a sampled parameter: it is the stored frame spacing, which is
        # the halved solver step times the stride.
        cols = np.stack([np.full(n, dt_ref * stride) if k == "dt"
                         else np.asarray(prm[k], dtype=np.float64).ravel()
                         for k in PARAM_COLUMNS], 1)
        dset = fh.create_dataset("params", data=cols)
        dset.attrs["columns"] = np.array(PARAM_COLUMNS, dtype="S")
        splits = fh.create_group("splits")
        splits.create_dataset("test", data=np.arange(n, dtype=np.int64))
        splits.create_dataset("train", data=np.zeros(0, dtype=np.int64))
        splits.create_dataset("val", data=np.zeros(0, dtype=np.int64))

        t0 = time.time()
        for start in range(0, n, chunk):
            idx = np.arange(start, min(start + chunk, n))
            blk = G._run_block(prm, idx, N, T_ref, dt_ref,
                               np.random.default_rng(seed + 1000 + start))
            C = blk["C"][:, ::stride].astype(np.float32)
            S = blk["src"][:, ::stride].astype(np.float32)
            for j, tid in enumerate(idx):
                g = fh.create_group(f"traj_{tid:04d}")
                g.create_dataset("C", data=C[j], compression="gzip", compression_opts=4)
                g.create_dataset("src", data=S[j], compression="gzip", compression_opts=4)
                g.create_dataset("qx", data=blk["qx"][j].astype(np.float32))
                g.create_dataset("qy", data=blk["qy"][j].astype(np.float32))
                g.create_dataset("D", data=blk["D"][j].astype(np.float32))
                g.attrs["x0"] = float(np.asarray(prm["x0"])[tid] / dx)
                g.attrs["y0"] = float(np.asarray(prm["y0"])[tid] / dx)
            print(f"  {start + idx.size}/{n} trajectories, {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {out} (N={N}, T={T}, dt={dt_ref * stride}, dx={dx})\n")
    return out


def evaluate(steps: int, cli_dx: float, out_path: str = OUT_H5, tag: str = "res128") -> dict:
    sys.path.insert(0, ".")
    from src.evaluate.evaluator import Evaluator            # noqa: E402

    os.makedirs(RES_DIR, exist_ok=True)
    out = {}
    for arm, run_dir in ARMS.items():
        ck = _ckpt(run_dir)
        ev = Evaluator(ck, data_path=out_path, steps=steps, device="cuda"
                       if os.environ.get("CUDA_OK") else "cpu")
        # dx is read from the checkpoint's config at construction; the transfer is
        # a change of dx, so it has to be pushed in afterwards.
        ev.dx = float(cli_dx)
        for mod in ev.model.modules():
            if hasattr(mod, "dx"):
                mod.dx = float(cli_dx)
        agg = ev.run("test", out_dir=None)
        agg.update({"arm": arm, "run_dir": run_dir, "transfer": "64->128",
                    "data": os.path.basename(out_path),
                    "refs": "60 fresh trajectories at 128^2"})
        out[arm] = agg
        json.dump(agg, open(os.path.join(RES_DIR, f"{tag}_{arm}.json"), "w"), indent=1)
        print(f"{arm:14s} rel_l2 {agg.get('rel_l2')}  r {agg.get('r')}  "
              f"w ratio {agg.get('width_ratio')}", flush=True)
    json.dump(out, open(os.path.join(RES_DIR, f"{tag}_all.json"), "w"), indent=1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="64^2 -> 128^2 transfer, no retraining")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--T-ref", type=int, default=400)
    ap.add_argument("--dt-ref", type=float, default=0.05)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", default=OUT_H5)
    a = ap.parse_args(argv)

    if a.generate:
        generate(a.n, a.N, a.T_ref, a.dt_ref, a.stride, a.seed, out=a.out)
    if a.evaluate:
        dx = 64.0 / a.N
        evaluate(a.steps, dx)
    if not (a.generate or a.evaluate):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
