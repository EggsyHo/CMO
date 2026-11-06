"""Training-entry gates (implementation.md step 9).

Three properties of `src/train.py` are load-bearing for a 22-run matrix and are
asserted here rather than eyeballed in a log:

1. an 8-window pass **runs and writes every artefact** (`config.yaml`, `run.json`,
   `stats.npz`, `best.pth`, `train_info.json`) -- a run that dies after `fit()` but
   before `json.dump` leaves a checkpoint with no provenance;
2. `stop_reason` is written, and the **budget** path is reachable (not just the
   patience/epochs paths), because the paper's cap is a wall-clock cap;
3. a run that produces non-finite loss reports `stop_reason == "nan"` instead of
   masquerading as early stopping with the best value of an earlier epoch.

Exit: 0 iff every gate passes.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_DIR = os.path.join(ROOT, "results", "_smoke")
_FAILED: list[str] = []


def gate(ok: bool, name: str, detail: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        _FAILED.append(name)
    return bool(ok)


def _entry():
    """`src/train.py` is shadowed by the `src/train/` package, so load by path."""
    spec = importlib.util.spec_from_file_location("train_entry",
                                                  os.path.join(ROOT, "src", "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-u", "src/train.py", *argv], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def _info(run_id: str = "cmo_adr2d_0") -> dict:
    path = os.path.join(SMOKE_DIR, run_id, "train_info.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
def check_smoke_run() -> None:
    """8 windows, one epoch: the step-9 acceptance run."""
    proc = _run(["--smoke", "8", "--set", "training.epochs=1"])
    tail = (proc.stdout or "").strip().splitlines()[-1:] or ["<no output>"]
    gate(proc.returncode == 0, "--smoke 8 --set training.epochs=1 exits 0", tail[-1])
    if proc.returncode != 0:
        print((proc.stderr or "")[-1500:])
        return

    info = _info()
    gate(info.get("stop_reason") == "epochs", "train_info.json carries stop_reason",
         f"stop_reason = {info.get('stop_reason')!r}, epochs = {info.get('epochs')}")
    gate(len(info.get("history", [])) == 1, "one epoch is one history record",
         f"{len(info.get('history', []))} record(s)")
    hist = (info.get("history") or [{}])[0]
    gate(all(isinstance(hist.get(k), (int, float)) and hist[k] == hist[k]
             for k in ("train_loss", "val_loss", "rel_l2_step1", "rel_l2_rollout", "rho")),
         "history records finite losses and both diagnostics",
         f"train={hist.get('train_loss'):.3e} step1={hist.get('rel_l2_step1'):.3e} "
         f"rollout={hist.get('rel_l2_rollout'):.3e} rho={hist.get('rho'):.3f}")

    # VRAM: `None` off CUDA on purpose. A reported 0.0 MB would silently satisfy
    # the 6 GB budget claim without measuring anything.
    dev, vram = info.get("device"), info.get("peak_vram_mb")
    gate((vram is None) if dev != "cuda" else (isinstance(vram, (int, float)) and vram > 0),
         "peak VRAM is reported only where it is measured",
         f"device = {dev}, peak_vram_mb = {vram!r}"
         + (" (no CUDA build: the 6 GB budget is unverified here)" if dev != "cuda" else ""))

    rdir = os.path.join(SMOKE_DIR, "cmo_adr2d_0")
    need = ["config.yaml", "run.json", "stats.npz", "best.pth", "train_info.json"]
    missing = [f for f in need if not os.path.exists(os.path.join(rdir, f))]
    gate(not missing, "every run artefact is written", f"missing: {missing or 'none'}")

    # the checkpoint must be self-sufficient: loadable into a freshly built model
    ck = torch.load(os.path.join(rdir, "best.pth"), map_location="cpu", weights_only=False)
    gate(ck.get("stop_reason") == info.get("stop_reason") and ck.get("model_name") == "cmo",
         "checkpoint repeats its own provenance",
         f"stop_reason = {ck.get('stop_reason')!r}, model_name = {ck.get('model_name')!r}")

    entry = _entry()
    cfg = entry.load_config(os.path.join(ROOT, "configs", "default.yaml"), [])
    normalizer = entry.Normalizer.load(os.path.join(rdir, "stats.npz"))
    model = entry.build_model(cfg, normalizer)
    try:
        model.load_state_dict(ck["model"], strict=True)
        detail = f"{len(ck['model'])} tensors, {int(model.n_params()):,} parameters"
        ok = True
    except Exception as exc:  # noqa: BLE001 - the failure being gated
        detail, ok = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300], False
    gate(ok, "checkpoint loads into a fresh build_model(strict=True)", detail)


def check_budget_stop() -> None:
    """A 0.3 s budget must stop with 'budget', not run 5 epochs."""
    proc = _run(["--smoke", "8", "--set", "training.epochs=5", "training.budget_min=0.005"])
    info = _info() if proc.returncode == 0 else {}
    gate(proc.returncode == 0 and info.get("stop_reason") == "budget",
         "a wall-clock budget ends the run with stop_reason='budget'",
         f"stop_reason = {info.get('stop_reason')!r} after {info.get('epochs')} epoch(s), "
         f"{info.get('wall_clock_s')} s")


def check_nan_stop() -> None:
    """A non-finite loss must be named, not mistaken for early stopping."""
    entry = _entry()
    cfg = entry.load_config(os.path.join(ROOT, "configs", "default.yaml"),
                            ["training.epochs=3", "training.budget_min=5"])

    class NaNModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, batch: dict) -> torch.Tensor:
            # keep the graph connected to `w`, else backward() fails before the
            # loss ever becomes NaN and the gate tests the wrong thing
            return batch["u"] * self.w * torch.full((), float("nan"))

    loader, val_loader, _, _ = entry.build_loaders(cfg, limit=8)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = entry.Trainer(NaNModel(), loader, val_loader, cfg, run_dir=tmp)
        info = trainer.fit()
    gate(info["stop_reason"] == "nan", "a non-finite loss stops with stop_reason='nan'",
         f"stop_reason = {info['stop_reason']!r} after {info['epochs']} epoch(s)"
         if info["stop_reason"] == "nan" else
         f"stop_reason = {info['stop_reason']!r} (NaN was swallowed as early stopping)")


def check_exit_contract() -> None:
    """§3.13 promises a non-zero exit on a non-finite `best_val`."""
    entry = _entry()
    good = entry.Trainer.fit
    entry.Trainer.fit = lambda self, *a, **k: {          # noqa: ARG005 - the stub
        "epochs": 0, "best_val": float("nan"), "stop_reason": "nan",
        "wall_clock_s": 0.0, "device": "cpu", "peak_vram_mb": None, "lr_final": 0.0}
    try:
        rc = entry.main(["--smoke", "8", "--set", "training.epochs=1"])
    finally:
        entry.Trainer.fit = good
    gate(rc == 1, "a non-finite best_val exits non-zero (so ablation.sh fails the cell)",
         f"main() returned {rc}")


def check_amp_autocast() -> None:
    """fp16 autocast must survive one forward+backward on the real model.

    This is not redundant with the smoke run: `amp` is silently inert on CPU
    (`self.amp` requires CUDA), so the fp16 path is *only* exercised on a machine
    with a GPU. It died there on `baddbmm_cuda not implemented for
    'ComplexHalf'` -- the spectral layer's complex einsum inside autocast.
    """
    if not torch.cuda.is_available():
        print("  SKIP  fp16 autocast forward+backward: no CUDA device in this venv")
        return
    entry = _entry()
    cfg = entry.load_config(os.path.join(ROOT, "configs", "default.yaml"), [])
    loader, _, train_ds, _ = entry.build_loaders(cfg, limit=8)
    normalizer = entry.load_or_fit_normalizer(train_ds, cfg)
    model = entry.build_model(cfg, normalizer).cuda().train()
    batch, _ = next(iter(loader))
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(batch)
            loss = out.float().pow(2).mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        ok = bool(torch.isfinite(loss)) and bool(grads) and all(
            bool(torch.isfinite(g).all()) for g in grads)
        detail = (f"loss = {float(loss):.3e}, {len(grads)} gradient tensors, "
                  f"peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    except Exception as exc:  # noqa: BLE001 - the failure being gated
        ok, detail = False, f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300]
    gate(ok, "fp16 autocast (training.amp) runs a forward+backward", detail)


def check_zero_target_sample() -> None:
    """An empty target field must be *dropped*, not floored to 1/eps.

    The real split has such windows (2 of 200 train windows have `u_{n+1} == 0`
    exactly). Under `den.clamp_min(1e-12)` they contribute `sqrt(num/1e-12)`,
    i.e. 1e6 x sqrt(num) -- and `sqrt`'s derivative at 0 is `inf`, so the sample
    with `num == 0` (the identity-at-init model on an empty field) turns every
    gradient in the batch into NaN.
    """
    entry = _entry()
    from src.train import CompositeLoss, rel_l2  # noqa: E402
    from src.models.blocks import MassPartition  # noqa: E402

    torch.manual_seed(0)
    pred = torch.randn(4, 3, 1, 16, 16, requires_grad=True)
    tgt = torch.randn(4, 3, 1, 16, 16)
    tgt[1, 1] = 0.0                              # the cold-start sample
    pred = pred.detach().requires_grad_(True)

    loss_fn = CompositeLoss(gamma1=1.0, gamma2=0.5, use_detail=True,
                            partition=MassPartition(16, 2))
    losses = loss_fn(pred, tgt)
    losses["total"].backward()
    grads = [p for p in [pred.grad] if p is not None]
    bad = sum(1 for g in grads if not bool(torch.isfinite(g).all()))

    valid = torch.stack([rel_l2(pred[i:i + 1, 1], tgt[i:i + 1, 1]) for i in (0, 2, 3)]).mean()
    excluded = torch.allclose(losses["l_one"], valid, rtol=1e-6)
    gate(bad == 0, "a zero-target sample leaves finite gradients",
         f"{bad} non-finite gradient tensor(s), l_one = {float(losses['l_one']):.4e}")
    gate(excluded, "a zero-target sample is excluded from the relative-L2 mean",
         f"l_one = {float(losses['l_one']):.4e} vs mean over the 3 valid samples "
         f"{float(valid):.4e} (floored would be ~1e6 x larger)")

    # second failure mode: an *exact* prediction (num == 0) puts the chain through
    # sqrt's infinite derivative. That is the state of the model at init on an
    # empty field, and it is what produced 25/25 NaN gradients on CUDA in fp32.
    perfect = tgt.clone().requires_grad_(True)
    losses2 = loss_fn(perfect, tgt)
    losses2["total"].backward()
    gate(bool(torch.isfinite(perfect.grad).all()),
         "an exact prediction (num == 0) leaves finite gradients",
         f"l_one = {float(losses2['l_one']):.4e}, "
         f"|grad| max = {float(perfect.grad.abs().max()):.3e}")


def main() -> int:
    check_smoke_run()
    check_budget_stop()
    check_nan_stop()
    check_exit_contract()
    check_amp_autocast()
    check_zero_target_sample()
    print(f"\n{len(_FAILED)} failed gate(s)" + ("" if _FAILED else " -- all pass"))
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
