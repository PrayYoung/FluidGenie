from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from tensorboardX import SummaryWriter
except Exception:  # pragma: no cover - optional at runtime
    SummaryWriter = None


def _to_float(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    return float(np.asarray(value))


def to_float_dict(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {k: _to_float(v) for k, v in metrics.items()}


class TrainingLogger:
    def __init__(
        self,
        out_dir: Path,
        run_name: str,
        log_every: int = 50,
        use_tb: bool = True,
    ):
        self.log_every = max(1, int(log_every))
        self.jsonl_path = out_dir / f"{run_name}_metrics.jsonl"
        self._tb = None
        self._start_time = time.time()

        if use_tb:
            if SummaryWriter is None:
                raise RuntimeError("TensorBoard logging requested but tensorboardX is not installed.")
            tb_dir = out_dir / "tb" / run_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb = SummaryWriter(logdir=str(tb_dir))

    def should_log(self, step: int) -> bool:
        return step % self.log_every == 0

    def log(self, step: int, metrics: Dict[str, Any], prefix: str = "train") -> None:
        m = to_float_dict(metrics)
        elapsed = max(1e-6, time.time() - self._start_time)
        m["elapsed_sec"] = elapsed
        record = {"step": int(step), **m}

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        msg_parts = [f"[{step}]"] + [f"{k}={v:.6f}" for k, v in m.items()]
        print(" ".join(msg_parts), flush=True)

        if self._tb is not None:
            for k, v in m.items():
                self._tb.add_scalar(f"{prefix}/{k}", v, step)
            self._tb.flush()

    def close(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
