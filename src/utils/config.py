"""YAML config loading with `--set a.b=c` overrides (implementation.md §3.17).

The config is returned as a nested `SimpleNamespace` so modules can write
`cfg.model.modes`; `to_dict()` gives back plain dicts for `yaml.dump` of the
*resolved* config (what makes "reproducible from one command" true).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import yaml

REQUIRED_KEYS = ("run.tag", "run.seed", "data.path", "model.name", "training.budget_min")

# Every arm/ablation yaml is a *delta* on the main config, not a standalone file.
# Without this merge, `data.epoch_windows` falls back to the code default (0 =
# "no cap"), so a baseline run quietly trains on all 55,160 windows per epoch
# instead of the 4,096 that the 25 min budget was measured against -- the first
# step-11 attempt did exactly that and never finished one epoch.
_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "default.yaml",
)


def _deep_merge(base: Mapping, override: Mapping) -> dict:
    out = copy.deepcopy(dict(base))
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _freeze(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return SimpleNamespace(**{str(k): _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_freeze(v) for v in obj]
    return obj


def _thaw(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _thaw(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_thaw(v) for v in obj]
    return obj


def _coerce(text: str) -> Any:
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text.strip().strip("'\"")


def _set_path(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def _get_path(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return None
        node = node[p]
    return node


def load_config(path: str, overrides: Iterable[str] | None = None) -> SimpleNamespace:
    """Load YAML as a delta on `configs/default.yaml`, apply overrides, validate, freeze."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}: {path}")
    cfg = copy.deepcopy(raw)
    if os.path.abspath(path) != os.path.abspath(_DEFAULT_CONFIG) and os.path.exists(_DEFAULT_CONFIG):
        with open(_DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
            cfg = _deep_merge(yaml.safe_load(fh) or {}, cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like a.b=value, got '{item}'")
        key, val = item.split("=", 1)
        _set_path(cfg, key.strip(), _coerce(val))
    missing = [k for k in REQUIRED_KEYS if _get_path(cfg, k) is None]
    if missing:
        raise KeyError(f"config {path} is missing required keys: {missing}")
    return _freeze(cfg)


def freeze(mapping: Mapping) -> SimpleNamespace:
    """`dict -> SimpleNamespace`, for rebuilding a config from a checkpoint."""
    return _freeze(mapping)


def to_dict(cfg: Any) -> dict:
    return _thaw(cfg)


def dump_config(cfg: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=True, allow_unicode=True)


def config_hash(cfg: Any) -> str:
    """Short hash of the resolved config, stored in every metrics file."""
    payload = json.dumps(to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """Attribute-path getter with default, e.g. get(cfg, "model.gate.beta_init", 4.0)."""
    node: Any = cfg
    for p in dotted.split("."):
        if isinstance(node, dict):
            if p not in node:
                return default
            node = node[p]
        else:
            if not hasattr(node, p):
                return default
            node = getattr(node, p)
    return node
