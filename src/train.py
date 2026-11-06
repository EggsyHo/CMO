"""Single entry point for one training run (implementation.md §3.13).

    python src/train.py --config configs/default.yaml --set run.seed=1

Everything a run needs to be reproducible lands in `results/{run_id}/`: the
**resolved** config (after `--set`), the normalizer statistics, `best.pth` and
`train_info.json`. `run_id` is derived from the config, so the 22 runs of §0.0
have predictable names and `ablation.sh` can be idempotent.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import numpy as np

import torch
import torch.utils.data as tud

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:                     # `python src/train.py` == `python -m src.train`
    sys.path.insert(0, ROOT)

from src.data.adr_dataset import ADRDataset, collate                     # noqa: E402
from src.data.transforms import Normalizer                               # noqa: E402
from src.models import build_model                                       # noqa: E402
from src.train import Trainer                                            # noqa: E402
from src.utils.config import config_hash, dump_config, get, load_config  # noqa: E402
from src.utils.logger import RunLogger                                   # noqa: E402
from src.utils.seed import set_seed                                      # noqa: E402


def build_run_id(cfg) -> str:
    """`{model}_{dataset}_{seed}` + the ablation tag when it says something new."""
    model = str(get(cfg, "model.name", "cmo"))
    dataset = os.path.splitext(os.path.basename(str(get(cfg, "data.path", "data"))))[0]
    seed = int(get(cfg, "run.seed", 0))
    run_id = f"{model}_{dataset}_{seed}"
    tag = str(get(cfg, "run.tag", "") or "")
    if tag and tag not in (model, dataset):
        run_id = f"{run_id}_{tag}"
    return run_id


def data_paths(cfg) -> list[str]:
    """`data.paths` (comma separated) wins over `data.path`; single file keeps the
    old behaviour bit for bit, so every published run is reproducible unchanged."""
    raw = str(get(cfg, "data.paths", "") or "").strip()
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    return [str(get(cfg, "data.path"))]


def load_or_fit_normalizer(dataset, cfg) -> Normalizer:
    """Fit once per (file(s), split, n_max) and share `stats.npz` across arms (§3.5).

    Shared on purpose: if each arm fitted its own statistics, an arm-to-arm
    difference could be a difference in input scaling. The provenance stamp carries
    the files' mtimes and `n_max`, so regenerated data refits instead of silently
    reusing statistics that no longer describe the fields they normalise.

    With several files the stamp covers **all** of them: a stamp keyed on the first
    file alone would keep reusing old statistics after one of the others was
    regenerated, which is the exact accident the provenance exists to prevent.
    """
    paths = data_paths(cfg)
    n_max = int(get(cfg, "data.norm_samples", 100))
    if len(paths) == 1:
        stats = os.path.join(os.path.dirname(os.path.abspath(paths[0])), "stats.npz")
        stamp = {"source": os.path.basename(paths[0]),
                 "mtime": int(os.path.getmtime(paths[0])), "n_max": n_max, "split": "train"}
    else:
        stats = os.path.join(os.path.dirname(os.path.abspath(paths[0])), "stats_mix.npz")
        stamp = {"source": "+".join(os.path.basename(p) for p in paths),
                 "mtime": max(int(os.path.getmtime(p)) for p in paths),
                 "n_max": n_max, "split": "train"}
    if os.path.exists(stats):
        prov = Normalizer.provenance(stats)
        if all(prov.get(k) == v for k, v in stamp.items()):
            return Normalizer.load(stats)
        print(f"[normalizer] {stats} does not match this dataset ({prov} vs {stamp}); refitting")
    normalizer = (dataset.fit_normalizer(n_max) if hasattr(dataset, "fit_normalizer")
                  else Normalizer.fit(dataset, n_max=n_max))
    os.makedirs(os.path.dirname(stats) or ".", exist_ok=True)
    normalizer.save(stats, provenance={**stamp, "n_traj": int(getattr(dataset, "n_traj", -1)),
                                       "n_index": int(len(dataset))})
    return normalizer


class GroupedBatchSampler(torch.utils.data.Sampler):
    """Batches drawn from **one** file each, files interleaved at random.

    `collate` requires a batch to share a single `dt` (the models take `dt` as a
    scalar), so mixing files inside a batch is not allowed. Mixing them across
    batches is exactly what a `dt`-randomised training set means here, and it keeps
    the model interface untouched.
    """

    def __init__(self, sizes, batch_size: int, n_batches: int, seed: int = 0):
        self.sizes = list(sizes)
        self.batch_size = int(batch_size)
        self.n_batches = int(n_batches)
        self.seed = int(seed)
        self.offsets = [0]
        for s in self.sizes:
            self.offsets.append(self.offsets[-1] + s)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_batches):
            f = int(rng.integers(len(self.sizes)))
            lo, hi = self.offsets[f], self.offsets[f + 1]
            yield rng.integers(lo, hi, size=min(self.batch_size, hi - lo)).tolist()


class StridedGroupedBatchSampler(torch.utils.data.Sampler):
    """The validation counterpart: fixed, evenly strided, still one file per batch.

    A plain `range(0, len, stride)` over the concatenation crosses the boundary
    between two files mid-batch, and `collate` rejects the mixed `dt` — this is not a
    theoretical risk, it is what the first real run did.
    """

    def __init__(self, sizes, batch_size: int, n_max: int = 0):
        self.batch_size = int(batch_size)
        self.batches = []
        off = 0
        per = (n_max // max(len(sizes), 1)) if n_max else 0
        for s in sizes:
            idx = list(range(off, off + s))
            if 0 < per < s:
                idx = idx[::max(s // per, 1)][:per]
            self.batches.extend(idx[i:i + self.batch_size]
                                for i in range(0, len(idx), self.batch_size))
            off += s

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def _build_mixed_loaders(cfg, paths, K, bs, workers, epoch_n, val_n, limit=None):
    """Several files, one `dt` each: batches are drawn per file (see GroupedBatchSampler).

    Why not per-sample `dt`: every model takes `dt` as a scalar and `collate` refuses a
    mixed batch, so the interface would have to change in five models to gain nothing —
    a batch still advances by one `dt`. Grouping by file gives the same training
    distribution with no model change, and keeps the single-file path untouched.
    """
    tr_sets = [ADRDataset(p, "train", K=K) for p in paths]
    va_sets = [ADRDataset(p, "val", K=K) for p in paths]
    tr_ds = tud.ConcatDataset(tr_sets)
    va_ds = tud.ConcatDataset(va_sets)
    pin = torch.cuda.is_available()
    if limit:
        idx = list(range(min(int(limit), len(tr_ds))))
        return (tud.DataLoader(tr_ds, sampler=tud.SubsetRandomSampler(idx), batch_size=bs,
                               collate_fn=collate, num_workers=workers, pin_memory=pin,
                               drop_last=False),
                tud.DataLoader(va_ds, sampler=idx[:min(len(idx), len(va_ds))], batch_size=bs,
                               collate_fn=collate, num_workers=workers, pin_memory=pin,
                               drop_last=False),
                tr_ds, va_ds)
    want = min(epoch_n, len(tr_ds)) if epoch_n else len(tr_ds)
    sampler = GroupedBatchSampler([len(d) for d in tr_sets], bs, max(want // bs, 1),
                                  seed=int(get(cfg, "run.seed", 0)))
    va_sampler = StridedGroupedBatchSampler([len(d) for d in va_sets], bs, val_n)
    return (tud.DataLoader(tr_ds, batch_sampler=sampler, collate_fn=collate,
                           num_workers=workers, pin_memory=pin,
                           persistent_workers=workers > 0),
            tud.DataLoader(va_ds, batch_sampler=va_sampler, collate_fn=collate,
                           num_workers=workers, pin_memory=pin,
                           persistent_workers=workers > 0),
            tr_ds, va_ds)


def build_loaders(cfg, limit: int | None = None):
    """`limit` subsets the *loaders* only: the datasets stay whole so
    `fit_normalizer` and `n_traj` still describe the real file.

    `data.epoch_windows` / `data.val_windows` cap a pass. They exist because the
    windows are stride-1: 280 trajectories x 197 windows means window i and i+1
    share 3 of their 4 frames, so a 55k-window epoch is ~200x redundant and costs
    25 min of wall clock to deliver one gradient step per trajectory-block. A
    capped pass draws a fresh random subset every epoch (`RandomSampler` without
    replacement), so over the run the model still sees the whole file; validation
    uses a fixed stride so epoch-to-epoch numbers stay comparable.
    """
    paths = data_paths(cfg)
    K = int(get(cfg, "data.K", 4))
    bs = int(get(cfg, "data.batch_size", 16))
    workers = int(get(cfg, "data.num_workers", 0))
    epoch_n = int(get(cfg, "data.epoch_windows", 0) or 0)
    val_n = int(get(cfg, "data.val_windows", 0) or 0)
    if len(paths) > 1:
        return _build_mixed_loaders(cfg, paths, K, bs, workers, epoch_n, val_n, limit)
    path = paths[0]
    train_ds = ADRDataset(path, "train", K=K)
    val_ds = ADRDataset(path, "val", K=K)
    common = dict(batch_size=bs, collate_fn=collate, num_workers=workers,
                  pin_memory=torch.cuda.is_available(),
                  persistent_workers=workers > 0)
    if limit:
        idx = list(range(min(int(limit), len(train_ds))))
        return (tud.DataLoader(train_ds, sampler=tud.SubsetRandomSampler(idx),
                               drop_last=False, **common),
                tud.DataLoader(val_ds, sampler=idx[:min(len(idx), len(val_ds))],
                               drop_last=False, **common),
                train_ds, val_ds)
    tr_sampler = (tud.RandomSampler(train_ds, num_samples=min(epoch_n, len(train_ds)),
                                    replacement=False)
                  if 0 < epoch_n < len(train_ds) else None)
    if 0 < val_n < len(val_ds):
        stride = max(len(val_ds) // val_n, 1)
        va_idx = list(range(0, len(val_ds), stride))[:val_n]
        va_sampler = va_idx
    else:
        va_sampler = None
    return (tud.DataLoader(train_ds, sampler=tr_sampler, shuffle=tr_sampler is None,
                           drop_last=False, **common),
            tud.DataLoader(val_ds, sampler=va_sampler, shuffle=False,
                           drop_last=False, **common),
            train_ds, val_ds)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="train one arm of the ADR-2D comparison")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. run.seed=1")
    ap.add_argument("--gpu", type=int, default=None, help="CUDA index; ignored on CPU builds")
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="train/validate on the first N windows only, writing to "
                         "results/_smoke/ so the real matrix is not polluted")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    if args.smoke:
        cfg.run.out = os.path.join(str(get(cfg, "run.out", "results")), "_smoke")
    set_seed(int(get(cfg, "run.seed", 0)))

    train_loader, val_loader, train_ds, _ = build_loaders(cfg, args.smoke)
    if len(train_ds) == 0:
        raise RuntimeError(f"empty training split in {get(cfg, 'data.path')}")
    normalizer = load_or_fit_normalizer(train_ds, cfg)

    model = build_model(cfg, normalizer)
    run_id = build_run_id(cfg)
    run_dir = os.path.join(str(get(cfg, "run.out", "results")), run_id)
    os.makedirs(run_dir, exist_ok=True)
    cfg.run.run_id = run_id
    dump_config(cfg, os.path.join(run_dir, "config.yaml"))
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as fh:
        json.dump({"run_id": run_id, "config": args.config, "overrides": args.set,
                   "config_hash": config_hash(cfg),
                   "param_count": int(model.n_params())}, fh, indent=2)

    init_from = get(cfg, "run.init_from", None)
    if init_from:
        model.load_state_dict(torch.load(init_from, map_location="cpu")["model"])

    logger = RunLogger(run_id, str(get(cfg, "logging.log_dir", "logs")))
    trainer = Trainer(model, train_loader, val_loader, cfg, logger,
                      run_dir=run_dir, normalizer=normalizer)
    info = trainer.fit()
    with open(os.path.join(run_dir, "train_info.json"), "w", encoding="utf-8") as fh:
        json.dump({"run_id": run_id, "param_count": int(model.n_params()),
                   "config_hash": config_hash(cfg), **info,
                   "history": trainer.history}, fh, indent=2)
    logger.close()
    print(f"[train] {run_id}: val={info['best_val']:.4e} "
          f"epochs={info['epochs']} stop={info['stop_reason']} "
          f"wall={info['wall_clock_s']:.0f}s -> {run_dir}", flush=True)

    # §3.13's exit contract, implemented rather than assumed: a diverged run or a
    # run that wrote no checkpoint must make `ablation.sh` fail on that cell
    # instead of silently recording a number.
    if not math.isfinite(float(info["best_val"])):
        print(f"[train] FAILED: best_val = {info['best_val']} is not finite", flush=True)
        return 1
    if not os.path.exists(os.path.join(run_dir, "best.pth")):
        print(f"[train] FAILED: no best.pth written to {run_dir}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
