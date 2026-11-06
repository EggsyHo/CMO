"""Three-term objective (implementation.md §3.11) — and nothing else.

No conservation penalty, no sorbed-phase term, no positivity barrier. Each of
those would turn "the operator conserves because of how it is built" into "the
operator conserves because it was paid to", which is the one claim the paper
cannot afford to lose. The only non-obvious term is the detail band, and it is
ablated (`ablation_no_detail`).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def sqrt_ratio(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """`mean(sqrt(num/den))` over the samples whose denominator is non-zero.

    Two guards, both forced by the data rather than by taste:

    * **`den == 0` must be dropped, not floored.** ADR windows whose second target
      row is still empty exist (2 of 200 train windows have `u_{n+1} == 0`
      exactly, because the source has not reached the domain yet). With
      `den.clamp_min(1e-12)` such a window contributes `sqrt(num/1e-12)` = 1e6 x
      sqrt(num): one cold-start sample hijacks the batch gradient, and the loss
      stops measuring the plume.
    * **`sqrt` has an infinite derivative at 0.** A sample with `num == 0` (an
      exact prediction, or the identity-at-init model on an empty field) yields
      `grad = inf`, and `2 x * inf = NaN` at `x = 0` poisons *every* parameter.
      Measured on CUDA in fp32: loss `5.9e-3` finite, 25 of 25 gradient tensors
      NaN. Clamping the ratio inside the sqrt bounds that derivative to
      `1/(2 sqrt(eps))` and costs nothing measurable.
    """
    valid = den > eps
    if not bool(valid.any()):
        return num.sum() * 0.0            # keep the graph: a 0 with a grad_fn
    return (num[valid] / den[valid]).clamp_min(eps).sqrt().mean()


def rel_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample `||p-t||/||t||`, averaged over the batch (torch twin of utils).

    Per-sample ratios are averaged instead of pooling the norms: a batch-pooled
    ratio is dominated by the samples with the most mass, and the small plumes are
    exactly the ones that matter at late time.
    """
    num = (pred - target).pow(2).flatten(1).sum(1)
    den = target.pow(2).flatten(1).sum(1)
    return sqrt_ratio(num, den, eps)


class CompositeLoss(nn.Module):
    def __init__(self, gamma1: float = 1.0, gamma2: float = 0.5,
                 use_detail: bool = True, partition=None, eps: float = 1e-12) -> None:
        super().__init__()
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2) if use_detail else 0.0
        self.use_detail = bool(use_detail)
        self.partition = partition
        self.eps = float(eps)

    def forward(self, pred_steps: torch.Tensor, target_steps: torch.Tensor) -> dict[str, torch.Tensor]:
        # [B, K+1, 1, N, N], step 0 is the shared input
        if pred_steps.dim() != 5 or target_steps.shape != pred_steps.shape:
            raise ValueError(f"expected [B,K+1,1,N,N] pairs, got "
                             f"{tuple(pred_steps.shape)} vs {tuple(target_steps.shape)}")
        kk = pred_steps.shape[1]
        if kk < 2:
            raise ValueError("need at least one predicted step (K >= 1)")

        l_one = rel_l2(pred_steps[:, 1], target_steps[:, 1])
        l_roll = torch.stack([rel_l2(pred_steps[:, k], target_steps[:, k])
                              for k in range(1, kk)]).mean()

        if self.use_detail and self.partition is not None:
            err = pred_steps[:, -1] - target_steps[:, -1]
            num = self.partition.residual(err).pow(2).flatten(1).sum(1)
            den = self.partition.residual(target_steps[:, -1]).pow(2).flatten(1).sum(1)
            l_detail = sqrt_ratio(num, den, self.eps)
        else:
            l_detail = pred_steps.new_zeros(())       # `use_detail=false` == gamma2 zeroed

        total = l_one + self.gamma1 * l_roll + self.gamma2 * l_detail
        return {"total": total, "l_one": l_one, "l_roll": l_roll, "l_detail": l_detail}
