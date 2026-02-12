from __future__ import annotations

from pathlib import Path
from typing import Any

from flax.serialization import from_bytes
from orbax.checkpoint import PyTreeCheckpointer
from orbax.checkpoint import utils as orbax_utils


def save_params(out_dir: Path, name: str, params: Any) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / name
    checkpointer = PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    checkpointer.save(ckpt_path, params, save_args=save_args, force=True)
    return ckpt_path


def load_params(ckpt_path: str, params_init: Any) -> Any:
    p = Path(ckpt_path)
    if p.exists() and p.is_dir():
        checkpointer = PyTreeCheckpointer()
        return checkpointer.restore(p, item=params_init)
    return from_bytes(params_init, p.read_bytes())
