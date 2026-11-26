"""
Train Latent Action Model (LAM) on fluid sequences.

Run:
  uv run python -m fluidgenie.training.train_lam \
    --data data/raw/ns2d \
    --out runs/lam \
    --steps 20000 \
    --seq-len 8
"""

from __future__ import annotations

import numpy as np
import jax
from fluidgenie.training.initialization import initialization_rngs
import jax.numpy as jnp
from flax.training import train_state
from tqdm import trange
from jaxtyping import Array, Float
import tyro

from configs.model_configs import LAMConfig
from fluidgenie.training.run_metadata import prepare_run
from fluidgenie.data.dataset_npz import create_grain_dataloader
from fluidgenie.training.codebook_restart import restart_codes, codebook_health
from fluidgenie.training.optimizers import make_optimizer
from fluidgenie.training.logging_utils import TrainingLogger
from fluidgenie.training.losses import lam_loss
from fluidgenie.training.checkpoint_utils import (
    save_params, save_training_state, restore_training_state,
)
from fluidgenie.models.lam import LatentActionModel


class TrainState(train_state.TrainState):
    pass


@jax.jit
def train_step(
    state: TrainState,
    batch: Float[Array, "b t h w c"],
    beta: float,
    dropout_key: jax.Array,
):
    def loss_fn(params):
        return lam_loss(state.apply_fn, params, batch, beta, dropout_key)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def main():
    args = tyro.cli(LAMConfig)

    rng = jax.random.PRNGKey(args.seed)

    stats_path = args.stats if args.stats else None
    loader = create_grain_dataloader(
        args.data,
        batch_size=args.batch,
        context=args.seq_len - 1,
        seed=args.seed,
        num_workers=args.grain_workers,
        stats_path=stats_path,
    )
    data_iter = iter(loader)
    x_ctx0, x_tgt0 = next(data_iter)
    batch0 = np.concatenate([x_ctx0, x_tgt0], axis=1)
    H, W, C = batch0.shape[-3:]

    model = LatentActionModel(
        in_dim=C,
        architecture_version=args.architecture_version,
        model_dim=args.model_dim,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        patch_size=args.patch_size,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        codebook_dropout=args.codebook_dropout,
    )

    params = model.init(
        initialization_rngs(rng), {"videos": jnp.array(batch0)}, training=True
    )["params"]
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=make_optimizer(args.lr, args.steps),
    )

    out = prepare_run(args.out, "lam", args, resume=args.resume)
    if args.stats:
        from pathlib import Path
        import shutil
        if Path(args.stats).resolve() != (out / "stats.npz").resolve():
            shutil.copy2(args.stats, out / "stats.npz")
    logger = TrainingLogger(
        out, run_name="lam", log_every=args.log_every, use_tb=bool(args.tb)
    )
    dropout_rng = jax.random.PRNGKey(args.seed + 1)

    usage = jnp.zeros((args.num_latents,), dtype=jnp.int32)
    if args.resume:
        state, dropout_rng, auxiliary = restore_training_state(
            out, state, dropout_rng, data_iter, auxiliary={"code_usage": usage},
        )
        usage = auxiliary["code_usage"]
    for step in trange(int(state.step), args.steps):
        x_ctx, x_tgt = next(data_iter)
        batch = jnp.array(np.concatenate([x_ctx, x_tgt], axis=1))
        dropout_rng, step_key = jax.random.split(dropout_rng)
        state, metrics = train_step(state, batch, args.vq_beta, step_key)
        counts = metrics.pop("code_usage")
        candidates = metrics.pop("restart_candidates")
        metrics.update(codebook_health(counts))
        usage = usage + counts
        if (
            args.restart_every > 0
            and (step + 1) % args.restart_every == 0
            and step + 1 <= args.restart_until_fraction * args.steps
        ):
            dropout_rng, restart_key = jax.random.split(dropout_rng)
            state, count = restart_codes(state, usage, candidates, restart_key)
            metrics["restarted_codes"] = count
            usage = jnp.zeros_like(usage)

        if logger.should_log(step) or "restarted_codes" in metrics:
            logger.log(step, metrics, prefix="train")
            # Optional small monitoring artifact, without changing training math.
            import os
            marker = os.environ.get("FLUIDGENIE_LAM_START_MARKER")
            if marker and not os.path.exists(marker):
                import json
                from pathlib import Path
                path = Path(marker)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"trainer": "lam", "training_step": int(state.step),
                                            "loss": float(metrics["loss"]), "state": "training_started"}) + "\n")

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0 and step > 0:
            save_params(out, f"step_{step:06d}", state.params)
            save_params(out, "latest", state.params)
            save_training_state(out, state, dropout_rng, auxiliary={"code_usage": usage})

    save_training_state(out, state, dropout_rng, auxiliary={"code_usage": usage})
    save_params(out, "final", state.params)
    save_params(out, "latest", state.params)
    logger.close()
    print("Saved LAM to", out)


if __name__ == "__main__":
    main()
