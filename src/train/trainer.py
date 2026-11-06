"""Wall-clock-budgeted trainer (implementation.md §3.12).

Two properties are load-bearing for the paper and are enforced here, not in prose:

* the training rollout is **teacher free** -- the model is fed its own prediction,
  so the rollout term is a rollout and not K independent one-steps;
* every run is bounded by `training.budget_min` and says so in its checkpoint
  (`stop_reason`), because the 22-run budget is a reported cap (Part 3 §3) and a
  budget-capped run must be identifiable from its own artefacts.
"""
from __future__ import annotations

import math
import os
import time

import torch

from ..models.blocks import MassPartition, detail_ratio
from ..utils.config import get, to_dict
from .losses import CompositeLoss, rel_l2


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v            # `dt` stays a float, metadata stays a list
    return out


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, logger=None,
                 run_dir: str | None = None, normalizer=None, device=None) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.normalizer = normalizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.K = int(get(cfg, "data.K", 4))
        self.budget_s = float(get(cfg, "training.budget_min", 25)) * 60.0
        self.max_epochs = int(get(cfg, "training.epochs", 200))
        self.patience = int(get(cfg, "training.patience", 8))
        self.grad_clip = float(get(cfg, "training.grad_clip", 1.0))
        self.amp = bool(get(cfg, "training.amp", True)) and self.device.type == "cuda"
        self.lr = float(get(cfg, "training.lr", 1e-3))
        self.min_lr_ratio = 1e-5 / max(self.lr, 1e-12)

        self.run_dir = run_dir or os.path.join(str(get(cfg, "run.out", "results")),
                                               str(get(cfg, "run.run_id", "run")))
        os.makedirs(self.run_dir, exist_ok=True)
        if normalizer is not None and hasattr(normalizer, "save"):
            normalizer.save(os.path.join(self.run_dir, "stats.npz"))

        self.model.to(self.device)
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                      weight_decay=float(get(cfg, "training.weight_decay", 0.0)))
        try:                                   # torch >= 2.4 spelling
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        block = int(get(cfg, "model.block", 2))
        self.loss_fn = CompositeLoss(
            gamma1=float(get(cfg, "training.gamma1", 1.0)),
            gamma2=float(get(cfg, "training.gamma2", 0.5)),
            use_detail=bool(get(cfg, "training.use_detail", True)),
            partition=MassPartition(int(get(cfg, "data.N", 64)), block),
        ).to(self.device)

        # initial horizon from the budget and the documented ~20 s/epoch; refined
        # after the first measured epoch. Starting at T_max=1 would decay the LR
        # to 1e-5 inside epoch 0 before any schedule estimate exists.
        guess_epochs = min(max(self.budget_s / 20.0, 1.0), 200.0)
        self._t_max = max(int(len(self.train_loader) * guess_epochs), 1)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_factor)
        self.best_val = float("inf")
        self._best_state: dict | None = None
        self._bad = 0
        self.step = 0
        self.epoch_s: float | None = None
        self.peak_vram_mb = 0.0
        self.history: list[dict] = []

    # ------------------------------------------------------------- schedule
    def _lr_factor(self, step: int) -> float:
        frac = min(max(step, 0) / max(self._t_max, 1), 1.0)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * frac))

    def _plan_schedule(self) -> None:
        """Cosine horizon from the *budget*, measured after the first epoch.

        The horizon is an estimate by construction (epoch time varies), so the
        realised final learning rate is logged: a run whose LR never reached
        `1e-5` must be visible as such, not silently averaged in.
        """
        if self.epoch_s is None:
            return
        expected = max(int((self.budget_s - self.epoch_s) / self.epoch_s), 1)
        steps_per_epoch = max(len(self.train_loader), 1)
        self._t_max = max(self.step + expected * steps_per_epoch, self.step + 1)

    # ------------------------------------------------------------------ step
    def _rollout(self, batch: dict, K: int) -> torch.Tensor:
        """Teacher-free: step k consumes the model's own output of step k-1."""
        preds = [batch["u"]]
        u = batch["u"]
        for k in range(K):
            src = batch["src_seq"][:, k] if "src_seq" in batch else batch["s"]
            u = self.model(dict(batch, u=u, s=src))
            preds.append(u)
        return torch.stack(preds, 1)          # [B, K+1, 1, N, N]

    def _targets(self, batch: dict, target: torch.Tensor) -> torch.Tensor:
        """[B, K+1]: the dataset already returns steps n..n+K with row 0 == `u`."""
        if target.shape[1] != self.K + 1:
            raise ValueError(f"target has {target.shape[1]} rows, expected K+1={self.K + 1}")
        return target

    def train_one_epoch(self) -> float:
        self.model.train()
        t0, total, n = time.time(), 0.0, 0
        for i, (batch, target) in enumerate(self.train_loader):
            batch = _to_device(batch, self.device)
            target = target.to(self.device, non_blocking=True)
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                pred = self._rollout(batch, self.K)
                losses = self.loss_fn(pred, self._targets(batch, target))
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.opt)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            self.step += 1
            total += float(losses["total"].detach())
            n += 1
            if i == 0:
                self._check_vram()
        self.epoch_s = time.time() - t0
        return total / max(n, 1)

    def _check_vram(self) -> None:
        if self.device.type != "cuda":
            return
        mb = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
        self.peak_vram_mb = max(self.peak_vram_mb, float(mb))
        if mb > 6144:
            print(f"[trainer] WARNING peak VRAM {mb:.0f} MB exceeds the 6 GB budget",
                  flush=True)

    # -------------------------------------------------------------- validate
    @torch.no_grad()
    def validate(self) -> tuple[float, dict]:
        self.model.eval()
        tot, n = 0.0, 0
        one, roll, rho = 0.0, 0.0, 0.0
        for batch, target in self.val_loader:
            batch = _to_device(batch, self.device)
            target = target.to(self.device, non_blocking=True)
            pred = self._rollout(batch, self.K)
            tgt = self._targets(batch, target)
            losses = self.loss_fn(pred, tgt)
            tot += float(losses["total"])
            one += float(rel_l2(pred[:, 1], tgt[:, 1]))
            roll += float(torch.stack([rel_l2(pred[:, k], tgt[:, k])
                                       for k in range(1, pred.shape[1])]).mean())
            rho += detail_ratio(pred[:, -1], int(get(self.cfg, "model.block", 2)))
            n += 1
        n = max(n, 1)
        return tot / n, {"rel_l2_step1": one / n, "rel_l2_rollout": roll / n, "rho": rho / n}

    # ------------------------------------------------------------------- fit
    def fit(self, max_epochs: int | None = None) -> dict:
        t0 = time.time()
        reason = "epochs"
        prev = 0.0
        for ep in range(int(max_epochs if max_epochs is not None else self.max_epochs)):
            tr = self.train_one_epoch()
            val, info = self.validate()
            elapsed = time.time() - t0
            per_epoch = elapsed - prev
            prev = elapsed
            rec = {"epoch": ep, "step": self.step, "train_loss": tr, "val_loss": val,
                   "lr": self.opt.param_groups[0]["lr"], "elapsed_s": round(elapsed, 1),
                   **info}
            self.history.append(rec)
            if self.logger is not None:
                self.logger.log(rec)
            # A diverged run must be *identifiable*, not reported as "patience"
            # with the best_val of three epochs ago. NaN never improves, so
            # without this branch it would silently look like early stopping.
            if not math.isfinite(tr) or not math.isfinite(val):
                reason = "nan"
                break
            if val < self.best_val - 1e-6:
                self.best_val, self._bad = val, 0
                # keep the weights, not just the number: the final save must write
                # the best epoch, and the last epoch's weights are not it
                self._best_state = {k: v.detach().clone()
                                    for k, v in self.model.state_dict().items()}
                self._save(reason="running", elapsed=elapsed)
            else:
                self._bad += 1
            # Checked *before* the next epoch, not after the last one. `elapsed >
            # budget_s` alone can only fire once an over-budget epoch has already
            # been paid for: the first real run paid 13 x 116 s = 1518 s (25.3 min)
            # against a declared 25 min ceiling. With one measured epoch in hand,
            # the next one is skipped when it would not fit inside a 5% margin --
            # so `wall_clock_s` stays under the declared cap instead of exceeding
            # it by "less than one epoch" and leaving the caption to explain.
            if elapsed > self.budget_s or elapsed + 1.05 * per_epoch > self.budget_s:
                reason = "budget"
                break
            if self._bad >= self.patience:
                reason = "patience"
                break
            self._plan_schedule()
        info = {"epochs": len(self.history), "best_val": self.best_val,
                "stop_reason": reason, "wall_clock_s": round(time.time() - t0, 1),
                # `None`, not 0.0, off CUDA: a reported 0 MB peak would read as a
                # measured number and silently satisfy the 6 GB budget claim.
                "device": self.device.type,
                "peak_vram_mb": round(self.peak_vram_mb, 1) if self.device.type == "cuda" else None,
                "lr_final": self.opt.param_groups[0]["lr"]}
        self._save(reason=reason, elapsed=time.time() - t0)
        return info

    def _save(self, reason: str, elapsed: float) -> None:
        state = self._best_state if self._best_state is not None else self.model.state_dict()
        torch.save({"model": state,
                    "cfg": to_dict(self.cfg),
                    "run_id": os.path.basename(self.run_dir),
                    "normalizer": os.path.join(self.run_dir, "stats.npz"),
                    "model_name": str(get(self.cfg, "model.name", "cmo")),
                    "epoch": len(self.history),
                    "val": self.best_val,
                    "stop_reason": reason,
                    "wall_clock_s": round(elapsed, 1)},
                   os.path.join(self.run_dir, "best.pth"))
