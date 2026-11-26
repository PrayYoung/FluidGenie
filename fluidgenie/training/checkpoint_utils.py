from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
from flax.serialization import from_bytes
from orbax.checkpoint import ArrayRestoreArgs, PyTreeCheckpointer


def save_params(out_dir: Path, name: str, params: Any) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = (out_dir / name).resolve()
    checkpointer = PyTreeCheckpointer()
    checkpointer.save(ckpt_path, params, force=True)
    return ckpt_path


def load_params(ckpt_path: str, params_init: Any) -> Any:
    p = Path(ckpt_path).expanduser().resolve()
    if p.exists() and p.is_dir():
        checkpointer = PyTreeCheckpointer()
        try:
            return checkpointer.restore(p, item=params_init)
        except ValueError as error:
            # A checkpoint written on CUDA may carry device-specific sharding
            # metadata. Fall back to the current local device for evaluation.
            if (
                "sharding" not in str(error).lower()
                and "cuda" not in str(error).lower()
            ):
                raise
            sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
            restore_args = jax.tree.map(
                lambda _leaf: ArrayRestoreArgs(sharding=sharding), params_init
            )
            return checkpointer.restore(p, item=params_init, restore_args=restore_args)
    return from_bytes(params_init, p.read_bytes())


def save_training_state(out_dir, state, rng, *, auxiliary=None):
    """Save optimizer moments, step, weights, and PRNG state for exact continuation."""
    return save_params(
        Path(out_dir),
        "train_state",
        {
            "params": state.params,
            "opt_state": state.opt_state,
            "step": state.step,
            "rng": rng,
            **({"auxiliary": auxiliary} if auxiliary is not None else {}),
        },
    )


def restore_training_state(out_dir, state, rng, data_iter, *, auxiliary=None):
    template = {
        "params": state.params,
        "opt_state": state.opt_state,
        "step": state.step,
        "rng": rng,
        **({"auxiliary": auxiliary} if auxiliary is not None else {}),
    }
    restored = load_params(str(Path(out_dir) / "train_state"), template)
    state = state.replace(
        params=restored["params"],
        opt_state=restored["opt_state"],
        step=restored["step"],
    )
    # The seeded Grain iterator has already consumed its shape-initialization batch.
    for _ in range(int(state.step)):
        next(data_iter)
    if auxiliary is not None:
        return state, restored["rng"], restored["auxiliary"]
    return state, restored["rng"]
