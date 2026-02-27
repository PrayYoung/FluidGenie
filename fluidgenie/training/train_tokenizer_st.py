"""
Train ST-VQVAE tokenizer (Jafar-style) on fluid sequences.

Run:
  uv run python -m fluidgenie.training.train_tokenizer_st \
    --data data/raw/ns2d \
    --out runs/tokenizer/st \
    --seq-len 4
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import trange
from jaxtyping import Array, Float
import tyro

from configs.model_configs import TokenizerConfig
from fluidgenie.data.dataset_npz import create_grain_dataloader
from fluidgenie.training.logging_utils import TrainingLogger
from fluidgenie.training.losses import tokenizer_st_loss
from fluidgenie.training.checkpoint_utils import save_params
from fluidgenie.models.tokenizer_st import TokenizerSTVQVAE


class TrainState(train_state.TrainState):
    pass


def make_train_step(alpha:float, beta: float, gamma: float):
    @jax.jit
    def _train_step(
        state: TrainState,
        batch: Float[Array, "b t h w c"],
        dropout_key: jax.Array,
    ):
        def loss_fn(params):
            return tokenizer_st_loss(state.apply_fn, params, batch, alpha, beta, gamma, dropout_key)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    return _train_step


def main():
    args = tyro.cli(TokenizerConfig)

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

    st_tokenizer_model = TokenizerSTVQVAE(
        in_dim=C,
        model_dim=args.model_dim,
        latent_dim=args.embed,
        num_latents=args.codebook,
        patch_size=args.patch_size,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        codebook_dropout=args.codebook_dropout,
    )

    st_tokenizer_params = st_tokenizer_model.init(
        rng, {"videos": jnp.array(batch0)}, training=True
    )["params"]
    state = TrainState.create(
        apply_fn=st_tokenizer_model.apply,
        params=st_tokenizer_params,
        tx=optax.adam(args.lr),
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(out, run_name="tokenizer_st", log_every=args.log_every, use_tb=bool(args.tb))
    dropout_rng = jax.random.PRNGKey(args.seed + 1)
    train_step = make_train_step(args.loss_alpha, args.loss_beta, args.loss_gamma)

    for step in trange(args.steps):
        x_ctx, x_tgt = next(data_iter)
        batch = jnp.array(np.concatenate([x_ctx, x_tgt], axis=1))
        dropout_rng, step_key = jax.random.split(dropout_rng)
        state, metrics = train_step(state, batch, step_key)

        if logger.should_log(step):
            logger.log(step, metrics, prefix="train")

        if step % 1000 == 0 and step > 0:
            save_params(out, f"step_{step:06d}", state.params)
            save_params(out, "latest", state.params)

    save_params(out, "final", state.params)
    save_params(out, "latest", state.params)
    logger.close()
    print("Saved ST tokenizer to", out)


if __name__ == "__main__":
    main()
