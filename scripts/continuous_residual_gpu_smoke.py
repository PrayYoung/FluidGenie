"""Exact-shape CUDA forward/loss/backprop gate for CR recurrent training."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tyro

from fluidgenie.models.continuous_residual import ContinuousResidualUNet
from fluidgenie.training.train_continuous_residual import residual_unroll_loss


@dataclass
class Config:
    unroll_steps: int = 4
    batch: int = 8
    stable: bool = True
    context: int = 2
    height: int = 64
    width: int = 64
    hidden_dims: tuple[int, int, int] = (64, 128, 256)


def make_gradient_gate(apply_fn):
    """Match the trainer's compiled execution, with dynamic minibatch inputs."""

    @jax.jit
    def gate(params, context, targets):
        def objective(p):
            loss, _ = residual_unroll_loss(apply_fn, p, context, targets)
            return loss

        return jax.value_and_grad(objective)(params)

    return gate


def main(cfg: Config) -> None:
    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"CUDA GPU required, got {devices}")
    context = jax.random.normal(
        jax.random.PRNGKey(0), (cfg.batch, cfg.context, cfg.height, cfg.width, 3)
    )
    targets = jax.random.normal(
        jax.random.PRNGKey(1), (cfg.batch, cfg.unroll_steps, cfg.height, cfg.width, 3)
    )
    model = ContinuousResidualUNet(
        channels=3,
        context_frames=cfg.context,
        hidden_dims=cfg.hidden_dims,
        num_res_blocks=2,
        stable=cfg.stable,
        residual_scales=(0.5, 0.5, 0.5),
    )
    params = model.init(jax.random.PRNGKey(2), context, training=True)["params"]

    gate = make_gradient_gate(model.apply)
    executable = gate.lower(params, context, targets).compile()
    planned = executable.memory_analysis()
    if planned is not None:
        print(
            {
                "compiled_memory": {
                    name: getattr(planned, name, None)
                    for name in (
                        "argument_size_in_bytes",
                        "output_size_in_bytes",
                        "temp_size_in_bytes",
                        "alias_size_in_bytes",
                    )
                }
            },
            flush=True,
        )
    loss, grads = executable(params, context, targets)
    leaves = jax.tree.leaves(grads)
    finite = bool(jnp.isfinite(loss)) and all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves
    )
    if not finite:
        raise RuntimeError("non-finite full-shape recurrent loss or gradients")
    jax.block_until_ready(loss)
    print(
        {
            "cuda_devices": [str(d) for d in devices],
            "execution": "jit_value_and_grad",
            "batch": cfg.batch,
            "resolution": [cfg.height, cfg.width],
            "hidden_dims": cfg.hidden_dims,
            "unroll_steps": cfg.unroll_steps,
            "loss": float(loss),
        }
    )
    print("CONTINUOUS_RESIDUAL_GPU_SMOKE_OK")


if __name__ == "__main__":
    main(tyro.cli(Config))
