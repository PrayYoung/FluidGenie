"""
Train Transformer dynamics in token space.

We assume you already trained a VQ tokenizer (VQVAE) and saved params ckpt.
This script:
  - loads NPZ episodes
  - samples windows of frames
  - encodes frames -> token grids via VQ encoder+codebook (argmin)
  - trains Transformer to predict next-frame tokens from context-frame tokens

Run (example):
  uv run python -m fluidgenie.training.train_dynamics \
    --data data/ns2d \
    --vq_ckpt runs/vq/latest \
    --out runs/dyn \
    --steps 20000 \
    --batch 4 \
    --context 2 \
    --codebook 512
"""

from __future__ import annotations

from configs.model_configs import DynamicsConfig
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import trange
from einops import rearrange
from jaxtyping import Array, Float, Int
import tyro

from fluidgenie.data.dataset_npz import NPZSequenceDataset, prefetch_iter
from fluidgenie.training.logging_utils import TrainingLogger
from fluidgenie.training.losses import dynamics_ar_loss
from fluidgenie.training.checkpoint_utils import save_params, load_params
from fluidgenie.models.vq_tokenizer import VQVAE, VQConfig
from fluidgenie.models.transformer_dynamics import TransformerDynamics, DynConfig


# -------------------------
# Data loader
# -------------------------

def infinite_loader(ds: NPZSequenceDataset, batch_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields:
      x_ctx: float32 [B, context, H, W, C]
      x_tgt: float32 [B, 1,       H, W, C]   (next frame)
    """
    idx = np.arange(len(ds))
    rng = np.random.default_rng(0)
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i : i + batch_size]
            xs, ys = [], []
            for j in batch_idx:
                x, y = ds[j]  # x:[context,H,W,C], y:[pred,H,W,C]
                xs.append(x)
                ys.append(y[:1])  # force pred=1
            yield np.stack(xs, axis=0), np.stack(ys, axis=0)


def flatten_token_sequence(tok_ctx: Int[Array, "b t h w"]) -> Int[Array, "b l"]:
    """
    tok_ctx: [B, context, h, w] -> [B, L] where L=context*h*w
    """
    return rearrange(tok_ctx, "b t h w -> b (t h w)")


def flatten_token_grid(tok: Int[Array, "b h w"]) -> Int[Array, "b l"]:
    """
    tok: [B, h, w] -> [B, h*w]
    """
    return rearrange(tok, "b h w -> b (h w)")


def encode_tokens_seq(
    vq_encode_fn, vq_params, x_seq: Float[Array, "b t h w c"]
) -> Int[Array, "b t h2 w2"]:
    """
    x_seq: [B, T, H, W, C] -> tok_seq: [B, T, h, w]
    """
    b, t, h, w, c = x_seq.shape
    x_flat = x_seq.reshape(b * t, h, w, c)
    tok_flat = vq_encode_fn(vq_params, x_flat)
    return tok_flat.reshape(b, t, tok_flat.shape[1], tok_flat.shape[2])


# -------------------------
# Model / training state
# -------------------------

class TrainState(train_state.TrainState):
    pass


def make_causal_mask(L: int) -> jnp.ndarray:
    """
    Standard causal mask for self-attention: [1, 1, L, L]
    """
    # Flax SelfAttention uses causal=True option instead of explicit mask; keep utility in case.
    m = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))
    return m[None, None, :, :]


def make_train_step(
    model_type: str,
    mask_token_id: int,
    mask_ratio_min: float,
    mask_ratio_max: float,
    mask_schedule: str,
    bos_token_id: int,
):
    @jax.jit
    def _train_step(
        state: TrainState,
        tok_in: Int[Array, "b l_in"],
        tok_tgt: Int[Array, "b l_out"],
        dropout_key: jax.Array,
        mask_key: jax.Array,
    ) -> Tuple[TrainState, dict]:
        """
        Args:
          tok_in:  int32 [B, L_in]  (context tokens flattened)
          tok_tgt: int32 [B, L_out] (next-frame tokens flattened)
        We train a model that outputs logits for each position in L_out, conditioned on tok_in.
        Simplest approach:
          feed [tok_in, tok_tgt_shifted] into one transformer and predict tok_tgt
        """
        B = tok_in.shape[0]
        L_in = tok_in.shape[1]
        L_out = tok_tgt.shape[1]

        # teacher forcing: shift target right with BOS token (assume token ids start at 0)
        bos = jnp.full((B, 1), bos_token_id, dtype=jnp.int32)
        tok_tgt_in = jnp.concatenate([bos, tok_tgt[:, :-1]], axis=1)  # [B, L_out]

        if model_type == "maskgit":
            # sample masking ratio
            t = jax.random.uniform(mask_key, shape=())
            if mask_schedule == "cosine":
                ratio = mask_ratio_min + (mask_ratio_max - mask_ratio_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))
            else:
                ratio = mask_ratio_min + (mask_ratio_max - mask_ratio_min) * t
            ratio = jnp.clip(ratio, 0.0, 1.0)

            # random mask per position
            scores = jax.random.uniform(mask_key, shape=(B, L_out))
            mask = scores < ratio
            # ensure at least one masked token per sample
            mask_any = jnp.any(mask, axis=1, keepdims=True)
            default_mask = jnp.concatenate([jnp.ones((B, 1), dtype=bool), jnp.zeros((B, L_out - 1), dtype=bool)], axis=1)
            mask = jnp.where(mask_any, mask, default_mask)

            tok_tgt_masked = jnp.where(mask, mask_token_id, tok_tgt)
            seq = jnp.concatenate([tok_in, tok_tgt_masked], axis=1)  # [B, L_in + L_out]
        else:
            seq = jnp.concatenate([tok_in, tok_tgt_in], axis=1)  # [B, L_in + L_out]

        def loss_fn(params):
            return dynamics_ar_loss(
                state.apply_fn,
                params,
                seq,
                tok_tgt,
                L_in,
                dropout_key,
                mask if model_type == "maskgit" else None,
            )

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    return _train_step




# -------------------------
# Main
# -------------------------

def main():
    args = tyro.cli(DynamicsConfig)

    rng = jax.random.PRNGKey(args.seed)

    # Dataset windows: context frames + 1 pred frame
    stats_path = args.stats if args.stats else None
    ds = NPZSequenceDataset(args.data, context=args.context, pred=1, stats_path=stats_path)
    loader = infinite_loader(ds, args.batch)
    if args.prefetch_batches > 0:
        loader = prefetch_iter(loader, prefetch=args.prefetch_batches, num_workers=args.prefetch_workers)

    # Infer shapes (H,W,C)
    x_ctx0, x_tgt0 = next(loader)
    H, W, C = x_ctx0.shape[-3:]
    x_ctx0 = jnp.array(x_ctx0)
    x_tgt0 = jnp.array(x_tgt0)

    # Load tokenizer params
    vq_cfg = VQConfig(codebook_size=args.codebook, embed_dim=args.embed, hidden=args.hidden)
    vq_model = VQVAE(vq_cfg, in_channels=C)
    vq_init = vq_model.init(rng, jnp.zeros((1, H, W, C), dtype=jnp.float32))["params"]
    vq_params = load_params(args.vq_ckpt, vq_init)

    # JIT encoder with model closed over (avoid passing Python objects into JIT)
    def _vq_encode_to_tokens(vq_params, x: jnp.ndarray) -> jnp.ndarray:
        _x_rec, tok, _commit, _cb = vq_model.apply({"params": vq_params}, x)
        return tok.astype(jnp.int32)

    vq_encode_to_tokens = jax.jit(_vq_encode_to_tokens)

    # Encode once to get token grid resolution
    tok_ctx0 = encode_tokens_seq(vq_encode_to_tokens, vq_params, x_ctx0)
    tok_tgt0 = encode_tokens_seq(vq_encode_to_tokens, vq_params, x_tgt0)[:, 0]

    h_tok, w_tok = tok_tgt0.shape[1], tok_tgt0.shape[2]
    L_in = args.context * h_tok * w_tok
    L_out = h_tok * w_tok
    max_len = L_in + L_out

    if args.model == "st_maskgit":
        raise ValueError("Use train_dynamics_st.py for model=st_maskgit.")

    vocab_size = args.codebook + (1 if args.model == "maskgit" else 0)
    dyn_cfg = DynConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_len=max_len,
    )
    dyn_model = TransformerDynamics(dyn_cfg)
    seq0 = jnp.zeros((1, max_len), dtype=jnp.int32)
    dyn_params = dyn_model.init(rng, seq0, train=True)["params"]

    state = TrainState.create(
        apply_fn=dyn_model.apply,
        params=dyn_params,
        tx=optax.adam(args.lr),
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(out, run_name="dynamics", log_every=args.log_every, use_tb=bool(args.tb))
    dropout_rng = jax.random.PRNGKey(args.seed + 1)
    mask_rng = jax.random.PRNGKey(args.seed + 2)

    train_step = make_train_step(
        args.model,
        mask_token_id=args.codebook,
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
        mask_schedule=args.mask_schedule,
        bos_token_id=args.bos_token_id,
    )

    # Train loop
    for step in trange(args.steps):
        x_ctx, x_tgt = next(loader)
        x_ctx = jnp.array(x_ctx)
        x_tgt = jnp.array(x_tgt)
        # Tokenize context and target (teacher forcing)
        tok_ctx = encode_tokens_seq(vq_encode_to_tokens, vq_params, x_ctx)
        tok_tgt = encode_tokens_seq(vq_encode_to_tokens, vq_params, x_tgt)[:, 0]

        tok_in = flatten_token_sequence(tok_ctx)     # [B, L_in]
        tok_out = flatten_token_grid(tok_tgt)        # [B, L_out]

        dropout_rng, step_key = jax.random.split(dropout_rng)
        mask_rng, mask_key = jax.random.split(mask_rng)
        state, metrics = train_step(state, tok_in, tok_out, step_key, mask_key)

        if logger.should_log(step):
            logger.log(step, metrics, prefix="train")

        if step % 1000 == 0 and step > 0:
            save_params(out, f"step_{step:06d}", state.params)
            save_params(out, "latest", state.params)

    save_params(out, "final", state.params)
    save_params(out, "latest", state.params)
    logger.close()
    print("Saved dynamics to", out)
    print(f"Token grid: {h_tok}x{w_tok}, L_in={L_in}, L_out={L_out}, max_len={max_len}")


if __name__ == "__main__":
    main()
