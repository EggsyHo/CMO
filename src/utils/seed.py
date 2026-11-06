"""Deterministic seeding for every entry point (implementation.md §3.17)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed python/numpy/torch and (optionally) force deterministic kernels.

    Determinism is worth the ~5% slowdown because the 2x2 interaction of
    Part 3 §2.2 is a difference of differences: noise in one cell propagates
    into the interaction term.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
