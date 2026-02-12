"""
Train VQ tokenizer on fluid frames (NPZ dataset).

This trains a frame-level VQ-VAE:
  field (H,W,C) -> discrete token grid -> reconstruction

Run:
  uv run python -m fluidgenie.training.train_tokenizer \
    --data data/ns2d_test \
    --out runs/vq \
    --steps 2000
"""

from __future__ import annotations

from configs.model_configs import TokenizerConfig
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import trange

from fluidgenie.data.dataset_npz import NPZSequenceDataset
from fluidgenie.training.logging_utils import TrainingLogger
from fluidgenie.training.losses import tokenizer_conv_loss
from fluidgenie.training.checkpoint_utils import save_params
from fluidgenie.models.vq_tokenizer import VQVAE, VQConfig
import tyro


# -------------------------
# Utilities
# -------------------------

def infinite_loader(ds: NPZSequenceDataset, batch_size: int) -> Iterator[np.ndarray]:
    idx = np.arange(len(ds))
    rng = np.random.default_rng(0)
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i : i + batch_size]
            frames = []
            for j in batch_idx:
                x, _ = ds[j]   # x: [context,H,W,C]
                # use the last context frame
                frames.append(x[-1])
            yield np.stack(frames, axis=0)


class TrainState(train_state.TrainState):
    pass


# -------------------------
# Training step
# -------------------------

def make_train_step(alpha: float, beta: float, gamma: float):
    @jax.jit
    def _train_step(state: TrainState, batch: jnp.ndarray):
        def loss_fn(params):
            return tokenizer_conv_loss(state.apply_fn, params, batch, alpha, beta, gamma)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    return _train_step


# -------------------------
# Main
# -------------------------

def main():
    args = tyro.cli(TokenizerConfig)

    rng = jax.random.PRNGKey(args.seed)

    # Dataset
    stats_path = args.stats if args.stats else None
    ds = NPZSequenceDataset(args.data, context=2, pred=1, stats_path=stats_path)
    if stats_path:
        stats = np.load(stats_path)
        mean = stats["mean"].reshape(1, 1, 1, -1).astype(np.float32)
        std = stats["std"].reshape(1, 1, 1, -1).astype(np.float32)
    else:
        mean = None
        std = None
    loader = infinite_loader(ds, args.batch)

    # Infer input channels
    sample_x, _ = ds[0]
    H, W, C = sample_x.shape[-3:]

    # Model
    cfg = VQConfig(
        codebook_size=args.codebook,
        embed_dim=args.embed,
        hidden=args.hidden,
    )
    model = VQVAE(cfg, in_channels=C)

    # Init
    batch0 = next(loader)
    params = model.init(rng, jnp.array(batch0))["params"]

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(args.lr),
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(out, run_name="tokenizer", log_every=args.log_every, use_tb=bool(args.tb))

    train_step = make_train_step(args.loss_alpha, args.loss_beta, args.loss_gamma)

    # Training loop
    for step in trange(args.steps):
        batch = next(loader)
        batch = jnp.array(batch)
        state, metrics = train_step(state, batch)

        if logger.should_log(step):
            logger.log(step, metrics, prefix="train")

        if step % 1000 == 0 and step > 0:
            save_params(out, f"step_{step:06d}", state.params)
            save_params(out, "latest", state.params)

        # quick recon snapshot
        if step % 500 == 0:
            x_rec, _, _, _ = model.apply({"params": state.params}, batch, mutable=False)
            x_rec_np = np.array(x_rec[0])  # first sample
            x_gt_np = np.array(batch[0])
            if mean is not None:
                x_rec_np = x_rec_np * (std[0, 0] + 1e-6) + mean[0, 0]
                x_gt_np = x_gt_np * (std[0, 0] + 1e-6) + mean[0, 0]

            snap = out / "snaps"
            snap.mkdir(exist_ok=True)
            np.save(snap / f"gt_{step:06d}.npy", x_gt_np)
            np.save(snap / f"rec_{step:06d}.npy", x_rec_np)

    # Save final
    save_params(out, "final", state.params)
    logger.close()
    print("Saved tokenizer to", out)


if __name__ == "__main__":
    main()
