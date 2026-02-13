"""
Train spatial-temporal MaskGIT dynamics (ST pipeline).

Run:
  uv run python -m fluidgenie.training.train_dynamics_st \
    --data data/raw/ns2d \
    --vq-ckpt runs/tokenizer/st/latest \
    --out runs/dynamics/st \
    --use-lam False
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import trange
from jaxtyping import Array, Float, Int
import tyro

from configs.model_configs import DynamicsConfig
from fluidgenie.data.dataset_npz import NPZSequenceDataset, create_grain_dataloader
from fluidgenie.training.logging_utils import TrainingLogger
from fluidgenie.training.losses import dynamics_st_loss
from fluidgenie.training.checkpoint_utils import save_params, load_params
from fluidgenie.models.base_tokenizer import VQVAE, VQConfig
from fluidgenie.models.dynamics_st import DynamicsSTMaskGIT
from fluidgenie.models.lam import LatentActionModel


def encode_tokens_seq(
    tokenizer_encode_fn, tokenizer_params, x_seq: Float[Array, "b t h w c"]
) -> Int[Array, "b t h2 w2"]:
    b, t, h, w, c = x_seq.shape
    x_flat = x_seq.reshape(b * t, h, w, c)
    tok_flat = tokenizer_encode_fn(tokenizer_params, x_flat)
    return tok_flat.reshape(b, t, tok_flat.shape[1], tok_flat.shape[2])


class TrainState(train_state.TrainState):
    pass


@jax.jit
def train_step_st(
    state: TrainState,
    tok_seq: Int[Array, "b t n"],
    mask_key: jax.Array,
    dropout_key: jax.Array,
    latent_actions: Float[Array, "b t m d"] | None = None,
) -> Tuple[TrainState, dict]:
    def loss_fn(params):
        return dynamics_st_loss(
            state.apply_fn,
            params,
            tok_seq,
            mask_key,
            dropout_key,
            latent_actions,
        )

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def main():
    args = tyro.cli(DynamicsConfig)
    if args.model != "st_maskgit":
        print(f"[warn] train_dynamics_st expects model=st_maskgit, got {args.model}. Continuing.")

    rng = jax.random.PRNGKey(args.seed)

    stats_path = args.stats if args.stats else None
    ds = NPZSequenceDataset(args.data, context=args.context, pred=1, stats_path=stats_path)
    loader = create_grain_dataloader(
        args.data,
        batch_size=args.batch,
        context=args.context,
        seed=args.seed,
        num_workers=args.grain_workers,
        stats_path=stats_path,
    )
    data_iter = iter(loader)

    x_ctx0, x_tgt0 = next(data_iter)
    H, W, C = x_ctx0.shape[-3:]
    x_ctx0 = jnp.array(x_ctx0)
    x_tgt0 = jnp.array(x_tgt0)

    vq_cfg = VQConfig(codebook_size=args.codebook, embed_dim=args.embed, hidden=args.hidden)
    base_tokenizer_model = VQVAE(vq_cfg, in_channels=C)
    base_tokenizer_init = base_tokenizer_model.init(
        rng, jnp.zeros((1, H, W, C), dtype=jnp.float32)
    )["params"]
    base_tokenizer_params = load_params(args.vq_ckpt, base_tokenizer_init)

    def _vq_encode_to_tokens(base_params, x: jnp.ndarray) -> jnp.ndarray:
        _x_rec, tok, _commit, _cb = base_tokenizer_model.apply({"params": base_params}, x)
        return tok.astype(jnp.int32)

    vq_encode_to_tokens = jax.jit(_vq_encode_to_tokens)

    lam_model = None
    lam_params = None
    lam_encode = None
    if args.use_lam:
        if not args.lam_ckpt:
            raise ValueError("--lam_ckpt is required when use_lam=True")
        lam_model = LatentActionModel(
            in_dim=C,
            model_dim=args.lam_model_dim,
            latent_dim=args.lam_latent_dim,
            num_latents=args.lam_num_latents,
            patch_size=args.lam_patch_size,
            num_blocks=args.lam_num_blocks,
            num_heads=args.lam_num_heads,
            dropout=args.lam_dropout,
            codebook_dropout=args.lam_codebook_dropout,
        )
        lam_init = lam_model.init(
            rng,
            {"videos": jnp.zeros((1, args.context + 1, H, W, C), dtype=jnp.float32)},
            training=False,
        )["params"]
        lam_params = load_params(args.lam_ckpt, lam_init)

        def _lam_encode(params, x_seq: jnp.ndarray) -> jnp.ndarray:
            out = lam_model.apply(
                {"params": params},
                x_seq,
                training=False,
                method=LatentActionModel.vq_encode,
            )
            return out["z_q"]

        lam_encode = jax.jit(_lam_encode)

    st_dynamics_model = DynamicsSTMaskGIT(
        model_dim=args.d_model,
        num_latents=args.codebook,
        num_blocks=args.n_layers,
        num_heads=args.n_heads,
        dropout=args.dropout,
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
    )
    tok_seq0 = encode_tokens_seq(
        vq_encode_to_tokens, base_tokenizer_params, jnp.concatenate([x_ctx0, x_tgt0], axis=1)
    )
    st_dynamics_params = st_dynamics_model.init(
        rng, {"video_tokens": tok_seq0, "mask_rng": rng}, training=True
    )["params"]

    state = TrainState.create(
        apply_fn=st_dynamics_model.apply,
        params=st_dynamics_params,
        tx=optax.adam(args.lr),
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(out, run_name="dynamics_st", log_every=args.log_every, use_tb=bool(args.tb))
    dropout_rng = jax.random.PRNGKey(args.seed + 1)
    mask_rng = jax.random.PRNGKey(args.seed + 2)

    for step in trange(args.steps):
        x_ctx, x_tgt = next(data_iter)
        x_ctx = jnp.array(x_ctx)
        x_tgt = jnp.array(x_tgt)

        x_seq = jnp.concatenate([x_ctx, x_tgt], axis=1)
        tok_seq = encode_tokens_seq(vq_encode_to_tokens, base_tokenizer_params, x_seq)
        latent_actions = None
        if args.use_lam:
            latent_actions = lam_encode(lam_params, x_seq)

        dropout_rng, step_key = jax.random.split(dropout_rng)
        mask_rng, mask_key = jax.random.split(mask_rng)
        state, metrics = train_step_st(state, tok_seq, mask_key, step_key, latent_actions)

        if logger.should_log(step):
            logger.log(step, metrics, prefix="train")

        if step % 1000 == 0 and step > 0:
            save_params(out, f"step_{step:06d}", state.params)
            save_params(out, "latest", state.params)

    save_params(out, "final", state.params)
    save_params(out, "latest", state.params)
    logger.close()
    print("Saved ST dynamics to", out)


if __name__ == "__main__":
    main()
