"""Training: objective (`losses`) and loop (`trainer`). No model, no metrics."""
from .losses import CompositeLoss, rel_l2
from .trainer import Trainer

__all__ = ["CompositeLoss", "rel_l2", "Trainer"]
