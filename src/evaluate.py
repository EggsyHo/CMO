"""Evaluate one checkpoint (implementation.md §3.15).

    python src/evaluate.py --ckpt results/cmo_adr2d_0/best.pth --modes test ext

Writes `metrics_{mode}.json`, `audit_{mode}.csv` and `rollout_{mode}.npz` into
the checkpoint's own directory, so the numbers live next to the weights they
belong to and a table row can always be traced back to a single file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluate import MODES, Evaluator   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="evaluate one trained arm")
    ap.add_argument("--ckpt", required=True, help="path to best.pth")
    ap.add_argument("--modes", nargs="+", default=["test"], choices=MODES)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--max-traj", type=int, default=None)
    ap.add_argument("--data", default=None, help="override data.path (debug on a smoke file)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="default: the checkpoint's directory")
    args = ap.parse_args(argv)

    out_dir = args.out or os.path.dirname(os.path.abspath(args.ckpt))
    ev = Evaluator(args.ckpt, data_path=args.data, steps=args.steps, device=args.device)
    summary = {}
    for mode in args.modes:
        try:
            res = ev.run(mode, out_dir=out_dir, max_traj=args.max_traj)
        except FileNotFoundError as exc:          # zero-shot files are optional
            print(f"[evaluate] skip {mode}: {exc}", flush=True)
            summary[mode] = {"skipped": str(exc)}
            continue
        # The whole result, not a hand-picked subset: step 17 requires a table row
        # that matches `metrics_{mode}.json` field for field, and a whitelist here
        # is how a new metric silently fails to reach `summary.csv`.
        summary[mode] = res
        print(f"[evaluate] {ev.run_id} {mode}: rel_l2_final={res['rel_l2']['final']:.4e} "
              f"r_final={res['conservation']['r_final_mean']:.3e} "
              f"w/w_ref={res['sharpness']['w_ratio_last']:.3f} "
              f"(med {res['sharpness']['w_ratio_median']:.3f}) "
              f"tv={res['tv_ratio']['mean']:.3f}", flush=True)
    # `out_dir` is created by `Evaluator._write`, which never runs when every mode is
    # skipped -- so a missing optional dataset used to crash here with
    # FileNotFoundError instead of writing the `skipped` record it had just built.
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
