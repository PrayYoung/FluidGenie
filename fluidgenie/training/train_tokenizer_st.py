"""
Train ST-VQVAE tokenizer (Jafar-style) on fluid sequences.

Run:
  uv run python -m fluidgenie.training.train_tokenizer_st \
    --data data/ns2d \
    --out runs/vq_st \
    --seq-len 4
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import trange
from jaxtyping import Array, Float
import tyro

from configs.model_configs import TokenizerConfig
from fluidgenie.data.dataset_npz import NPZSequenceDataset, prefetch_iter
from fluidgenie.training.logging_utils import TrainingLogger
from fluidgenie.training.losses import tokenizer_st_loss
from fluidgenie.training.checkpoint_utils import save_params
from fluidgenie.models.tokenizer_st import TokenizerSTVQVAE


def infinite_loader(ds: NPZSequenceDataset, batch_size: int) -> Iterator[np.ndarray]:
    idx = np.arange(len(ds))
    rng = np.random.default_rng(0)
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i : i + batch_size]
            seqs = []
            for j in batch_idx:
                x, y = ds[j]
                seqs.append(np.concatenate([x, y], axis=0))
            yield np.stack(seqs, axis=0)


class TrainState(train_state.TrainState):
    pass


def make_train_step(beta: float):
    @jax.jit
    def _train_step(
        state: TrainState,
        batch: Float[Array, "b t h w c"],
        dropout_key: jax.Array,
    ):
        def loss_fn(params):
            return tokenizer_st_loss(state.apply_fn, params, batch, beta, dropout_key)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    return _train_step


def main():
    args = tyro.cli(TokenizerConfig)

    rng = jax.random.PRNGKey(args.seed)

    stats_path = args.stats if args.stats else None
    ds = NPZSequenceDataset(args.data, context=args.seq_len - 1, pred=1, stats_path=stats_path)
    loader = infinite_loader(ds, args.batch)
    if args.prefetch_batches > 0:
        loader = prefetch_iter(loader, prefetch=args.prefetch_batches, num_workers=args.prefetch_workers)

    sample_x, sample_y = ds[0]
    H, W, C = sample_x.shape[-3:]

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
        positional=args.st_positional,
    )

    batch0 = next(loader)
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
    train_step = make_train_step(args.loss_beta)

    for step in trange(args.steps):
        batch = next(loader)
        batch = jnp.array(batch)
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
