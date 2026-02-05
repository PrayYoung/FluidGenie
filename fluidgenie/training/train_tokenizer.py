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

import argparse
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.serialization import to_bytes
from tqdm import trange

from fluidgenie.data.dataset_npz import NPZSequenceDataset
from fluidgenie.training.logging_utils import TrainingLogger
from fluidgenie.models.vq_tokenizer import VQVAE, VQConfig


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

@jax.jit
def train_step(state: TrainState, batch: jnp.ndarray):
    def loss_fn(params):
        x_rec, _, commit, codebook = state.apply_fn({"params": params}, batch)
        recon = jnp.mean((x_rec - batch) ** 2)
        loss = recon + commit + codebook
        return loss, {
            "loss": loss,
            "recon": recon,
            "commit": commit,
            "codebook": codebook,
        }

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--codebook", type=int, default=1024)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--tb", type=int, default=1, help="1=write TensorBoard logs, 0=disable")
    args = ap.parse_args()

    rng = jax.random.PRNGKey(args.seed)

    # Dataset
    ds = NPZSequenceDataset(args.data, context=2, pred=1)
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

    # Training loop
    for step in trange(args.steps):
        batch = next(loader)
        batch = jnp.array(batch)
        state, metrics = train_step(state, batch)

        if logger.should_log(step):
            logger.log(step, metrics, prefix="train")

        if step % 1000 == 0 and step > 0:
            ckpt = out / f"step_{step:06d}.ckpt"
            ckpt.write_bytes(to_bytes(state.params))
            (out / "latest.ckpt").write_bytes(to_bytes(state.params))

        # quick recon snapshot
        if step % 500 == 0:
            x_rec, _, _, _ = model.apply({"params": state.params}, batch, mutable=False)
            x_rec_np = np.array(x_rec[0])  # first sample
            x_gt_np = np.array(batch[0])

            snap = out / "snaps"
            snap.mkdir(exist_ok=True)
            np.save(snap / f"gt_{step:06d}.npy", x_gt_np)
            np.save(snap / f"rec_{step:06d}.npy", x_rec_np)

    # Save final
    (out / "final.ckpt").write_bytes(to_bytes(state.params))
    logger.close()
    print("Saved tokenizer to", out)


if __name__ == "__main__":
    main()
