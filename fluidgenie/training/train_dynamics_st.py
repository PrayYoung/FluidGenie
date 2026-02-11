"""
Train spatial-temporal MaskGIT dynamics (ST pipeline).

Run:
  uv run python -m fluidgenie.training.train_dynamics_st \
    --data data/ns2d \
    --vq-ckpt runs/vq/latest.ckpt \
    --out runs/dyn_st \
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
from flax.serialization import from_bytes, to_bytes
from tqdm import trange
import tyro

from configs.model_configs import DynamicsConfig
from fluidgenie.data.dataset_npz import NPZSequenceDataset
from fluidgenie.training.logging_utils import TrainingLogger
from fluidgenie.models.vq_tokenizer import VQVAE, VQConfig
from fluidgenie.models.dynamics_st import DynamicsSTMaskGIT
from fluidgenie.models.lam import LatentActionModel


def infinite_loader(ds: NPZSequenceDataset, batch_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(ds))
    rng = np.random.default_rng(0)
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i : i + batch_size]
            xs, ys = [], []
            for j in batch_idx:
                x, y = ds[j]
                xs.append(x)
                ys.append(y[:1])
            yield np.stack(xs, axis=0), np.stack(ys, axis=0)


def encode_tokens_seq(vq_encode_fn, vq_params, x_seq: jnp.ndarray) -> jnp.ndarray:
    b, t, h, w, c = x_seq.shape
    x_flat = x_seq.reshape(b * t, h, w, c)
    tok_flat = vq_encode_fn(vq_params, x_flat)
    return tok_flat.reshape(b, t, tok_flat.shape[1], tok_flat.shape[2])


class TrainState(train_state.TrainState):
    pass


@jax.jit
def train_step_st(
    state: TrainState,
    tok_seq: jnp.ndarray,
    mask_key: jnp.ndarray,
    dropout_key: jnp.ndarray,
    latent_actions: jnp.ndarray | None = None,
) -> Tuple[TrainState, dict]:
    def loss_fn(params):
        batch = {"video_tokens": tok_seq, "mask_rng": mask_key}
        if latent_actions is not None:
            batch["latent_actions"] = latent_actions
        outputs = state.apply_fn(
            {"params": params},
            batch,
            training=True,
            rngs={"dropout": dropout_key},
        )
        mask = outputs["mask"].astype(jnp.float32)
        logits = outputs["token_logits"]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, tok_seq)
        denom = jnp.maximum(mask.sum(), 1.0)
        loss = (mask * ce).sum() / denom
        acc = (mask * (logits.argmax(-1) == tok_seq)).sum() / denom
        return loss, {"loss": loss, "masked_acc": acc}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def main():
    args = tyro.cli(DynamicsConfig)

    rng = jax.random.PRNGKey(args.seed)

    stats_path = args.stats if args.stats else None
    ds = NPZSequenceDataset(args.data, context=args.context, pred=1, stats_path=stats_path)
    loader = infinite_loader(ds, args.batch)

    x_ctx0, x_tgt0 = next(loader)
    H, W, C = x_ctx0.shape[-3:]
    x_ctx0 = jnp.array(x_ctx0)
    x_tgt0 = jnp.array(x_tgt0)

    vq_cfg = VQConfig(codebook_size=args.codebook, embed_dim=args.embed, hidden=args.hidden)
    vq_model = VQVAE(vq_cfg, in_channels=C)
    vq_init = vq_model.init(rng, jnp.zeros((1, H, W, C), dtype=jnp.float32))["params"]
    vq_params = from_bytes(vq_init, Path(args.vq_ckpt).read_bytes())

    def _vq_encode_to_tokens(vq_params, x: jnp.ndarray) -> jnp.ndarray:
        _x_rec, tok, _commit, _cb = vq_model.apply({"params": vq_params}, x)
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
        lam_params = from_bytes(lam_init, Path(args.lam_ckpt).read_bytes())

        def _lam_encode(params, x_seq: jnp.ndarray) -> jnp.ndarray:
            out = lam_model.apply(
                {"params": params},
                x_seq,
                training=False,
                method=LatentActionModel.vq_encode,
            )
            return out["z_q"]

        lam_encode = jax.jit(_lam_encode)

    dyn_model = DynamicsSTMaskGIT(
        model_dim=args.d_model,
        num_latents=args.codebook,
        num_blocks=args.n_layers,
        num_heads=args.n_heads,
        dropout=args.dropout,
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
    )
    tok_seq0 = encode_tokens_seq(vq_encode_to_tokens, vq_params, jnp.concatenate([x_ctx0, x_tgt0], axis=1))
    dyn_params = dyn_model.init(rng, {"video_tokens": tok_seq0, "mask_rng": rng}, training=True)["params"]

    state = TrainState.create(
        apply_fn=dyn_model.apply,
        params=dyn_params,
        tx=optax.adam(args.lr),
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(out, run_name="dynamics_st", log_every=args.log_every, use_tb=bool(args.tb))
    dropout_rng = jax.random.PRNGKey(args.seed + 1)
    mask_rng = jax.random.PRNGKey(args.seed + 2)

    for step in trange(args.steps):
        x_ctx, x_tgt = next(loader)
        x_ctx = jnp.array(x_ctx)
        x_tgt = jnp.array(x_tgt)

        x_seq = jnp.concatenate([x_ctx, x_tgt], axis=1)
        tok_seq = encode_tokens_seq(vq_encode_to_tokens, vq_params, x_seq)
        latent_actions = None
        if args.use_lam:
            latent_actions = lam_encode(lam_params, x_seq)

        dropout_rng, step_key = jax.random.split(dropout_rng)
        mask_rng, mask_key = jax.random.split(mask_rng)
        state, metrics = train_step_st(state, tok_seq, mask_key, step_key, latent_actions)

        if logger.should_log(step):
            logger.log(step, metrics, prefix="train")

        if step % 1000 == 0 and step > 0:
            (out / f"step_{step:06d}.ckpt").write_bytes(to_bytes(state.params))
            (out / "latest.ckpt").write_bytes(to_bytes(state.params))

    (out / "final.ckpt").write_bytes(to_bytes(state.params))
    (out / "latest.ckpt").write_bytes(to_bytes(state.params))
    logger.close()
    print("Saved ST dynamics to", out)


if __name__ == "__main__":
    main()
