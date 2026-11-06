"""Step-6 data contract gate: HDF5 -> batch dict -> Normalizer round trip.

Run:  python tests/check_data.py [--data data/adr2d] [--k 4] [--n-max 100] [--no-write]

Every check prints the number it gates on, because these are the numbers the
data section of the paper inherits. Gates (non-zero exit on any failure):

  G1 splits      train/val/test disjoint and covering, except that a zero-shot
                 file carries one set under both `val` and `test` (it has no
                 training split, so it cannot be resplit into train/val/test).
  G2 index       every window has K+1 frames and the count is exactly
                 sum_traj (T_i - K + 1).  `range(T_i - K)` is one short per
                 trajectory: it drops the *last* step of every case, which is the
                 step a long-horizon test reads.
  G3 batch       exactly the §2 keys (plus the documented `src_seq`/`traj_id`),
                 the §2 shapes/dtypes, and `target[:, k] == C[step + k]` read
                 straight from HDF5 -- an off-by-one here would train the model to
                 predict the state it was given.
  G4 supports    Pe / Da / kappa / theta / rho_b / smax per regime block, plus the
                 *derived* |q| and R0 = 1 + rho_b smax K_L / theta. Checking the
                 derived pair is what catches a reformulated lambda or K_L, which a
                 range check on the sampled columns cannot see.
  G5 normalizer  denormalize(normalize(u)) relative error < 1e-5 for u>0, the
                 log-space increment matches its analytic Jacobian, and the five
                 physical coefficients pass through untouched.
  G6 stats.npz   frozen from the *train* split only, provenance-stamped, reload is
                 bit-identical (`stats.npz` is the file every arm shares).
  G7 guards      regime='train' cannot enter ZeroShotDataset, and an empty train
                 split cannot enter ADRDataset(split='train').
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.adr_dataset import ADRDataset, ZeroShotDataset, collate  # noqa: E402
from src.data.transforms import Normalizer                              # noqa: E402

# §2 batch contract: key -> (shape with B/N/K symbolic, dtype)
REQUIRED = {
    "u":      (("B", 1, "N", "N"), torch.float32),
    "q":      (("B", 2, "N", "N"), torch.float32),
    "D":      (("B", 3, "N", "N"), torch.float32),
    "s":      (("B", 1, "N", "N"), torch.float32),
    "theta":  (("B", 1, 1, 1), torch.float32),
    "rho_b":  (("B", 1, 1, 1), torch.float32),
    "smax":   (("B", 1, 1, 1), torch.float32),
    "k_lang": (("B", 1, 1, 1), torch.float32),
    "lam":    (("B", 1, 1, 1), torch.float32),
}
# documented extensions: `src_seq` feeds the K-step rollout (§3.13), `traj_id` is
# provenance for the audit table. Anything else is an undocumented key -> fail.
ALLOWED_EXTRA = {"src_seq": ("B", "K", 1, "N", "N"), "traj_id": None, "dt": None}

# §3.1 bands, keyed by /params column. `q` and `R0` are *not* columns: they are
# recomputed from (lam, theta, Da) and (rho_b, smax, k_lang, theta) so that the
# check fails if the lambda / K_L formulas are ever rewritten.
BANDS = {
    "train":   {"Pe": (2.0, 15.0), "Da": (0.0, 3.0), "kappa": (1.0, 12.0),
                "theta": (0.25, 0.40), "rho_b": (1.2, 1.8), "smax": (1.0, 10.0),
                "q": (0.05, 0.35), "R0": (1.5, 8.0)},
    "ext":     {"Pe": (1.0, 1.8), "Da": (3.5, 5.0), "kappa": (14.0, 20.0),
                "theta": (0.25, 0.40), "rho_b": (1.2, 1.8), "smax": (1.0, 10.0),
                "q": (0.40, 0.60), "R0": (9.0, 12.0)},
    "kappa1":  {"q": (0.18, 0.22), "R0": (4.0, 5.0), "Da": (1.4, 1.6),
                "kappa": (1.0, 1.0), "theta": (0.30, 0.34),
                "rho_b": (1.4, 1.6), "smax": (4.0, 6.0)},
    "kappa20": {"q": (0.18, 0.22), "R0": (4.0, 5.0), "Da": (1.4, 1.6),
                "kappa": (20.0, 20.0), "theta": (0.30, 0.34),
                "rho_b": (1.4, 1.6), "smax": (4.0, 6.0)},
}
DOMAIN_L = 64.0

FAILS: list[str] = []


def gate(ok: bool, label: str, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: {detail}")
    if not ok:
        FAILS.append(f"{label}: {detail}")


def _decode(seq) -> list[str]:
    return [c.decode() if isinstance(c, (bytes, np.bytes_)) else str(c) for c in seq]


# --------------------------------------------------------------------- G1
def check_splits(path: str) -> dict:
    with h5py.File(path, "r") as f:
        meta = dict(f["/meta"].attrs)
        trajs = sorted(k for k in f if k.startswith("traj_"))
        sp = {s: np.asarray(f[f"/splits/{s}"][:], dtype=np.int64) for s in ("train", "val", "test")}
        T = {t: int(f[f"/{t}/C"].shape[0]) - 1 for t in trajs}
        C_shape = tuple(f[f"/{trajs[0]}/C"].shape)
    n = len(trajs)
    name = os.path.basename(path)
    gate(n == int(meta.get("n_traj", -1)), f"{name} n_traj",
         f"{n} trajectories, meta.n_traj={meta.get('n_traj')}")
    if sp["train"].size == 0:
        # zero-shot file: one set, exposed under both names (Evaluator reads `val`)
        gate(np.array_equal(np.sort(sp["val"]), np.arange(n))
             and np.array_equal(np.sort(sp["test"]), np.arange(n)),
             f"{name} single zero-shot set",
             f"train empty; val={sp['val'].size} test={sp['test'].size} ids=0..{n - 1} "
             "(one set under two names, so nothing can split it)")
    else:
        ids = np.concatenate([sp["train"], sp["val"], sp["test"]])
        gate(ids.size == n and np.array_equal(np.sort(ids), np.arange(n)),
             f"{name} split partition",
             f"{sp['train'].size}/{sp['val'].size}/{sp['test'].size} cover each trajectory exactly once")
    gate(len(set(T.values())) == 1, f"{name} uniform horizon",
         f"T={sorted(set(T.values()))} C={C_shape} dt={meta.get('dt')} regime={meta.get('regime')}")
    return {"meta": meta, "splits": sp, "T": T, "n": n}


# --------------------------------------------------------------------- G2
def check_index(ds: ADRDataset, info: dict) -> None:
    K = ds.K
    expect = sum(int(info["T"][ds.traj_names[t]]) - K + 1 for t in ds.split_ids)
    gate(len(ds) == expect, f"{os.path.basename(ds.path)}[{ds.split}] index size",
         f"len={len(ds)}, sum(T-K+1)={expect} (K={K})")
    steps: dict[int, list[int]] = {}
    for tid, n in ds.index:
        steps.setdefault(tid, []).append(n)
    bad = [t for t, ns in steps.items()
           if sorted(ns) != list(range(min(ns), max(ns) + 1)) or min(ns) != 0
           or max(ns) != info["T"][ds.traj_names[t]] - K]
    gate(not bad, f"{os.path.basename(ds.path)}[{ds.split}] windows",
         f"{len(steps)} trajectories, steps 0..T-K contiguous (offenders={len(bad)})")


# --------------------------------------------------------------------- G3
def check_batch(ds: ADRDataset, info: dict) -> dict:
    K, N = ds.K, ds.N
    # a post-injection window: step 0 of every trajectory is identically zero, and
    # a zero batch satisfies every shape assertion while testing nothing
    tid = int(ds.split_ids[0])
    last = info["T"][ds.traj_names[tid]] - K
    picks = [i for i, (t, n) in enumerate(ds.index) if t == tid and n == min(last, 150)][:1]
    picks += [i for i, (t, n) in enumerate(ds.index) if t == tid and n == last][:1]
    batch, target = collate([ds[i] for i in picks])
    B = len(picks)

    keys = set(batch)
    gate(keys == set(REQUIRED) | set(ALLOWED_EXTRA), "batch keys", f"{sorted(keys)}")
    for k, (shape, dtype) in REQUIRED.items():
        want = tuple(B if s == "B" else N if s == "N" else s for s in shape)
        got, dt = tuple(batch[k].shape), batch[k].dtype
        gate(got == want and dt == dtype, f"batch[{k}]", f"shape={got} dtype={dt}")
    gate(tuple(batch["src_seq"].shape) == (B, K, 1, N, N), "batch[src_seq]",
         f"shape={tuple(batch['src_seq'].shape)}")
    gate(tuple(target.shape) == (B, K + 1, 1, N, N), "batch[target]",
         f"shape={tuple(target.shape)} dtype={target.dtype} (row 0 aliases u)")
    gate(isinstance(batch["dt"], float), "batch[dt]",
         f"python {type(batch['dt']).__name__} = {batch['dt']}")

    # alignment, read straight from HDF5 (not through the dataset)
    with h5py.File(ds.path, "r") as f:
        g = f[f"/{ds.traj_names[tid]}"]
        ok_u = ok_t = ok_s = True
        for j, i in enumerate(picks):
            _, n = ds.index[i]
            C = np.asarray(g["C"][n: n + K + 1], np.float32)
            src = np.asarray(g["src"][n: n + K], np.float32)
            ok_u &= np.array_equal(batch["u"][j, 0].numpy(), C[0])
            ok_t &= np.array_equal(target[j, :, 0].numpy(), C)
            ok_s &= np.array_equal(batch["s"][j, 0].numpy(), src[0])
            ok_s &= np.array_equal(batch["src_seq"][j, :, 0].numpy(), src)
    gate(ok_u, "u == C[step]", f"window steps {[ds.index[i][1] for i in picks]}")
    gate(ok_t, "target == C[step : step+K+1]",
         "bit-identical (float32; the data path never rescales)")
    gate(ok_s, "s == src[step], src_seq == src[step : step+K]", "bit-identical")

    # scalars must be the trajectory's own parameters; the batch is float32, so the
    # tolerance is float32 round-off (1e-12 would fail on a correct file)
    prm = ds.params
    worst = max(abs(float(batch[k][j]) - prm[k][tid]) / max(abs(prm[k][tid]), 1e-30)
                for k in ("theta", "rho_b", "smax", "k_lang", "lam") for j in range(B))
    gate(worst < 1e-6, "scalars == /params[row]",
         f"traj {tid}, max relative diff {worst:.1e} (float32 store), theta={float(batch['theta'][0]):.4f}")
    gate(float(batch["u"].min()) >= 0.0, "u >= 0", f"min={float(batch['u'].min()):.2e}")
    gate(float(batch["u"].max()) > 0.0, "u non-trivial",
         f"max={float(batch['u'].max()):.3e} mg/L")
    return batch


# --------------------------------------------------------------------- G4
def check_supports(path: str, info: dict) -> None:
    with h5py.File(path, "r") as f:
        arr = np.asarray(f["/params"][:], np.float64)
        cols = _decode(f["/params"].attrs["columns"])
    col = {c: arr[:, i] for i, c in enumerate(cols)}
    name = os.path.basename(path)
    raw = info["meta"].get("regime_blocks")
    blocks = [(str(b[0]), int(b[1])) for b in json.loads(raw)] if raw else \
             [(str(info["meta"].get("regime")), len(arr))]
    i = 0
    for regime, n in blocks:
        seg = {c: v[i: i + n] for c, v in col.items()}
        i += n
        with np.errstate(divide="ignore", invalid="ignore"):
            seg["q"] = seg["lam"] * DOMAIN_L * seg["theta"] / np.where(seg["Da"] > 1e-9, seg["Da"], np.nan)
        seg["R0"] = 1.0 + seg["rho_b"] * seg["smax"] * seg["k_lang"] / seg["theta"]
        for key, (lo, hi) in BANDS.get(regime, {}).items():
            v = seg[key][np.isfinite(seg[key])]
            if v.size == 0:
                gate(False, f"{name}[{regime}] {key}", "no finite values")
                continue
            tol = 1e-9 + 1e-7 * max(abs(lo), abs(hi))
            gate(float(v.min()) >= lo - tol and float(v.max()) <= hi + tol,
                 f"{name}[{regime}] {key}",
                 f"[{v.min():.5g}, {v.max():.5g}] in [{lo:g}, {hi:g}]"
                 + ("" if v.size == n else f" (n={v.size}/{n}, Da=0 skipped)"))
    if len(blocks) == 2 and blocks[0][0] == "train" and blocks[1][0] == "ext":
        # "excluded by construction": the only statement that survives inspection
        # is that the *dimensionless groups* the solver feels do not overlap
        for key in ("Pe", "Da", "kappa"):
            a = BANDS["train"][key]
            b = BANDS["ext"][key]
            gate(a[1] < b[0] or b[1] < a[0], f"{name} {key} disjoint",
                 f"train [{a[0]:g}, {a[1]:g}] vs ext [{b[0]:g}, {b[1]:g}]")
    if "k_lang" in col and "lam" in col:
        print(f"  note  {name}: lam [{col['lam'].min():.4g}, {col['lam'].max():.4g}] 1/s, "
              f"k_lang [{col['k_lang'].min():.4g}, {col['k_lang'].max():.4g}] L/mg (derived, not sampled)")


# --------------------------------------------------------------------- G5
class _PrefixView:
    """The first `n` samples of a dataset: the fit bug the strided fit exists to avoid."""

    def __init__(self, ds, n: int) -> None:
        self.ds, self.n = ds, n

    def __len__(self) -> int:
        return min(self.n, len(self.ds))

    def __getitem__(self, i: int):
        return self.ds[i]


def check_normalizer(ds: ADRDataset, batch: dict, n_max: int) -> Normalizer:
    norm = Normalizer.fit(ds, n_max=n_max)
    print(f"  fit on {min(len(ds), n_max)} strided train windows: "
          + "  ".join(f"std[{k}]={np.max(norm.std[k]):.3e}" for k in ("u", "s", "q", "D")))
    floor = min(float(np.min(v)) for v in norm.std.values())
    gate(floor > 1e-6, "std floor", f"min std = {floor:.3e} (> 1e-6, so no channel is constant)")

    # The absolute std is not gated: the plume covers ~5% of the cells, so the
    # log-space std is small by construction and any threshold would be a guess.
    # What *is* gated is that the shipped 100-sample estimate is not a narrow
    # slice: it must agree with a 10x denser strided estimate (§3.5).
    dense = Normalizer.fit(ds, n_max=10 * n_max)
    ratio = float(np.max(dense.std["u"]) / np.max(norm.std["u"]))
    gate(0.5 < ratio < 2.0, "fit is a converged estimate",
         f"std_u {float(np.max(norm.std['u'])):.3e} (n_max={n_max}) vs "
         f"{float(np.max(dense.std['u'])):.3e} (n_max={10 * n_max}), ratio {ratio:.2f} in [0.5, 2]")
    prefix = Normalizer.fit(_PrefixView(ds, n_max), n_max=n_max)
    print(f"  note  prefix fit (one trajectory's first {n_max} steps): std_u="
          f"{float(np.max(prefix.std['u'])):.2e}, i.e. "
          f"{float(np.max(norm.std['u']) / np.max(prefix.std['u'])):.1f}x narrower than strided")

    u = batch["u"]
    z = norm.normalize_u(u)
    back = norm.denormalize_u(z)
    peak = float(u.max())
    # float32 gives absolute, not relative, invertibility: `z*s + m` cancels the
    # low bits of log1p(u), so a ghost cell at 1e-9 of the peak comes back as 0.
    # Both statements are gated, because only the first one matters for the model.
    bulk = u >= 1e-4 * peak
    rel_bulk = float(((back - u).abs() / u.clamp_min(1e-30))[bulk].max())
    abs_over_peak = float((back - u).abs().max()) / peak
    zero = torch.zeros(1, 1, ds.N, ds.N)
    zero_abs = float(norm.denormalize_u(norm.normalize_u(zero)).abs().max())

    # and the map itself, in float64: if this is not exact, the transform is wrong
    # (gated on the bulk too -- at u -> 0 the map is exact only up to the sign of
    # `z*s + m`, which is a statement about 1e-30 mg/L, not about the transform)
    u64 = u.double()
    m = np.asarray(norm.mean["u"], np.float64).reshape(1, -1, 1, 1)
    s = np.asarray(norm.std["u"], np.float64).reshape(1, -1, 1, 1)
    back64 = np.expm1(((np.log1p(u64.numpy()) - m) / s) * s + m)
    rel64 = float(np.max(np.abs(back64 - u64.numpy())[bulk.numpy()]
                         / np.maximum(u64.numpy()[bulk.numpy()], 1e-300)))
    clamped = (u > 0) & (back == 0)
    tail = int(clamped.sum())
    tail_peak = float(u[clamped].max()) / peak if tail else 0.0

    gate(float(torch.isfinite(z).all()) and float(z.abs().max()) < 50.0, "z finite and bounded",
         f"max|z(u)|={float(z.abs().max()):.2f} on a post-injection window")
    gate(rel64 < 1e-12, "transform invertible in float64 (bulk)", f"max relative error {rel64:.2e}")
    gate(rel_bulk < 1e-5, "normalize -> denormalize (bulk)",
         f"max relative error {rel_bulk:.2e} for u >= 1e-4*peak "
         f"({int(bulk.sum())}/{u.numel()} cells), gate 1e-5")
    gate(abs_over_peak < 1e-6, "round trip in absolute terms",
         f"max |back-u| = {abs_over_peak:.2e} * peak; the {tail} positive cells that "
         f"clamp to 0 are all <= {tail_peak:.1e} * peak")
    gate(zero_abs < 1e-6, "u == 0 stays 0", f"max |back| = {zero_abs:.2e} mg/L")

    # the increment map: du = (u+1) * std * dz, checked against a float64 finite
    # difference of the forward map (checking it in float32 differences would
    # measure the cancellation of two nearly equal float32 numbers, not the formula)
    dz = 1e-6
    z64 = (np.log1p(u64.numpy()) - m) / s
    du_true = np.expm1((z64[bulk.numpy()] + dz) * s + m) - np.expm1(z64[bulk.numpy()] * s + m)
    du_pred = norm.denormalize_delta(torch.full_like(u[bulk], dz), u[bulk]).double().numpy()
    rel_d = float(np.max(np.abs(du_pred - du_true) / np.maximum(np.abs(du_true), 1e-300)))
    gate(rel_d < 1e-5, "denormalize_delta vs analytic Jacobian",
         f"max relative error {rel_d:.2e} (dz={dz:g}, bulk cells)")

    # the five physical coefficients must not be rescaled: the balance module
    # consumes them, and a z-scored theta would silently change the budget
    ok = all(torch.equal(norm.normalize_params(k, batch[k]), batch[k])
             for k in ("theta", "rho_b", "smax", "k_lang", "lam"))
    gate(ok, "normalize_params identity", "theta, rho_b, smax, k_lang, lam returned unchanged")
    return norm


# --------------------------------------------------------------------- G6
def freeze_stats(norm: Normalizer, ds: ADRDataset, path: str, n_max: int, write: bool) -> None:
    stamp = {"source": os.path.basename(ds.path), "mtime": int(os.path.getmtime(ds.path)),
             "n_max": n_max, "split": "train", "n_traj": int(ds.n_traj),
             "n_index": int(len(ds))}
    if write:
        norm.save(path, provenance=stamp)
    if not os.path.exists(path):
        gate(False, "stats.npz present", f"missing: {path}")
        return
    gate(True, "stats.npz present", f"{path} ({os.path.getsize(path) / 2**10:.1f} kB)")
    got = Normalizer.provenance(path)
    gate(all(got.get(k) == v for k, v in stamp.items()), "stats.npz provenance", f"{got}")
    a = Normalizer.load(path)
    tmp = path + ".tmp.npz"
    a.save(tmp)
    b = Normalizer.load(tmp)
    same = (all(np.array_equal(a.mean[k], b.mean[k]) and np.array_equal(a.std[k], b.std[k])
                for k in a.mean) and a.log_fields == b.log_fields)
    gate(same, "stats.npz reload", f"save->load->save bit-identical, log_fields={sorted(a.log_fields)}")
    os.remove(tmp)
    probe = torch.ones(1, 1, 4, 4) * 0.3
    rt = float((norm.denormalize_u(norm.normalize_u(probe)) - 0.3).abs().max())
    gate(rt < 1e-5, "frozen stats reproduce the fit", f"|u_hat - 0.3| = {rt:.2e}")


# --------------------------------------------------------------------- G7
def check_guards(data_dir: str) -> None:
    ext = os.path.join(data_dir, "adr2d_ext.h5")
    if os.path.exists(ext):
        try:
            ADRDataset(ext, "train", K=1)
            gate(False, "empty train split rejected", "ADRDataset(ext, 'train') did not raise")
        except ValueError as exc:
            gate(True, "empty train split rejected", f"{type(exc).__name__}: {str(exc)[:60]}...")
        z = ZeroShotDataset(ext, K=1)
        gate(z.regime == "ext" and len(z) > 0, "ZeroShotDataset(ext)",
             f"len={len(z)} regime={z.regime} dt={z.dt}")
    try:
        ZeroShotDataset(os.path.join(data_dir, "adr2d.h5"), K=1)
        gate(False, "regime='train' rejected by ZeroShotDataset", "no exception")
    except ValueError as exc:
        gate(True, "regime='train' rejected by ZeroShotDataset", f"{type(exc).__name__}: {str(exc)[:50]}...")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/adr2d")
    ap.add_argument("--k", type=int, default=4, help="training rollout horizon K (§3.18)")
    ap.add_argument("--n-max", type=int, default=100, help="normalizer fit samples (§3.5)")
    ap.add_argument("--no-write", action="store_true", help="check stats.npz, do not write it")
    args = ap.parse_args(argv)

    main_h5 = os.path.join(args.data, "adr2d.h5")
    if not os.path.exists(main_h5):
        print(f"[skip] {main_h5} not found: generate the data first (scripts/gen_data.sh)")
        return 0

    print("[1/7] splits")
    infos = {}
    for name in ("adr2d.h5", "adr2d_ext.h5", "adr2d_kappa.h5", "adr2d_long.h5"):
        path = os.path.join(args.data, name)
        if os.path.exists(path):
            infos[name] = check_splits(path)

    print("[2/7] index")
    ds = ADRDataset(main_h5, "train", K=args.k)
    check_index(ds, infos["adr2d.h5"])
    for split in ("val", "test"):
        check_index(ADRDataset(main_h5, split, K=args.k), infos["adr2d.h5"])

    print("[3/7] batch contract")
    batch = check_batch(ds, infos["adr2d.h5"])

    print("[4/7] parameter supports")
    for name, info in infos.items():
        check_supports(os.path.join(args.data, name), info)

    print("[5/7] normalizer")
    norm = check_normalizer(ds, batch, args.n_max)

    print("[6/7] stats.npz")
    freeze_stats(norm, ds, os.path.join(args.data, "stats.npz"), args.n_max, not args.no_write)

    print("[7/7] zero-shot guards")
    check_guards(args.data)

    print()
    if FAILS:
        print(f"{len(FAILS)} gate(s) failed:")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
