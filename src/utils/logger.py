"""Append-only jsonl run logger with a compact console mirror (§3.17)."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # numpy scalars / torch scalars / anything with .item()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # pragma: no cover - defensive
            pass
    return str(obj)


class RunLogger:
    """One instance per run; no global state, so two runs can share a process."""

    def __init__(self, run_id: str, log_dir: str = "logs", echo: bool = True) -> None:
        self.run_id = run_id
        self.log_dir = log_dir
        self.echo = echo
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{run_id}.jsonl")
        self._t0 = time.time()

    def log(self, record: dict) -> None:
        row = {"run_id": self.run_id, "wall_clock_s": round(time.time() - self._t0, 3)}
        row.update(_jsonable(record))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.echo:
            keys = ("epoch", "val_loss", "loss", "stop_reason", "message")
            brief = " ".join(f"{k}={row[k]}" for k in keys if k in row)
            print(f"[{self.run_id}] {brief}", flush=True)

    def close(self) -> None:
        sys.stdout.flush()
