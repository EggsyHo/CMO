"""Baselines: identical batch contract, different structure (§3.9)."""
from .fluxnet import FluxNetBaseline
from .fno import FNO2dBaseline
from .nfv import NFVBaseline

__all__ = ["FNO2dBaseline", "FluxNetBaseline", "NFVBaseline"]
