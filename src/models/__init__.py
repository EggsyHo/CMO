"""Model registry (implementation.md §3.10).

Every entry takes `(cfg, normalizer)` and returns a module whose
`forward(batch, return_diag=False)` yields physical `[B,1,N,N]`. `train.py` and
`evaluate.py` therefore swap models by name only, and the mass audit is one
object applied to all four.
"""
from __future__ import annotations

import sys
from typing import Any

import torch.nn as nn

from .baseline.fluxnet import FluxNetBaseline
from .baseline.fno import FNO2dBaseline
from .baseline.nfv import NFVBaseline
from .cmo import CMO

MODELS = {
    "cmo": CMO,
    "fno": FNO2dBaseline,
    "fluxnet": FluxNetBaseline,
    "nfv": NFVBaseline,
}

# implementation.md §3.10: the budget row of the main table is only meaningful if
# the arms sit in the same range. This is a warning, not an error -- the 24-mode
# sweep and deliberately shrunken ablations must still be buildable.
PARAM_BUDGET = (0.15e6, 0.60e6)


def build_model(cfg: Any, normalizer: Any = None, verbose: bool = True) -> nn.Module:
    from ..utils.config import get

    name = str(get(cfg, "model.name", "cmo")).lower()
    if name not in MODELS:
        raise KeyError(f"unknown model.name='{name}'; choose from {sorted(MODELS)}")
    model = MODELS[name](cfg, normalizer)
    n = model.n_params()
    lo, hi = PARAM_BUDGET
    if not (lo <= n <= hi):
        print(f"[build_model] WARNING {name}: {n:,} params outside the "
              f"budget [{lo:,.0f}, {hi:,.0f}]", file=sys.stderr, flush=True)
    if verbose:
        print(f"[build_model] {name}: {n:,} parameters", flush=True)
    setattr(model, "param_count", n)
    return model


__all__ = ["MODELS", "PARAM_BUDGET", "build_model", "CMO", "FNO2dBaseline",
           "FluxNetBaseline", "NFVBaseline"]
