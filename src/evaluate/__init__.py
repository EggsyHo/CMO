"""Evaluation: free-running rollouts (`rollout`) and the metric table (`evaluator`)."""
from .evaluator import MODES, Evaluator
from .rollout import audit_series, rollout

__all__ = ["Evaluator", "MODES", "rollout", "audit_series"]
