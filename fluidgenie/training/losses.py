from __future__ import annotations

from typing import Tuple, Dict, Any, Callable

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float, Int


def spatial_grads(
    img: Float[Array, "b h w c"],
) -> Tuple[Float[Array, "b h w c"], Float[Array, "b h w c"]]:
    # Central differences with edge padding to keep H,W sizes
    img_pad = jnp.pad(img, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    dx = 0.5 * (img_pad[:, 1:-1, 2:, :] - img_pad[:, 1:-1, :-2, :])
    dy = 0.5 * (img_pad[:, 2:, 1:-1, :] - img_pad[:, :-2, 1:-1, :])
    return dx, dy


def compute_vorticity(img: Float[Array, "b h w c"]) -> Float[Array, "b h w"]:
    # img: [B,H,W,C], assumes C>=2 with u=0, v=1
    u = img[..., 0]
    v = img[..., 1]
    u_pad = jnp.pad(u, ((0, 0), (1, 1), (1, 1)), mode="edge")
    v_pad = jnp.pad(v, ((0, 0), (1, 1), (1, 1)), mode="edge")
    du_dy = 0.5 * (u_pad[:, 2:, 1:-1] - u_pad[:, :-2, 1:-1])
    dv_dx = 0.5 * (v_pad[:, 1:-1, 2:] - v_pad[:, 1:-1, :-2])
    return dv_dx - du_dy


def tokenizer_conv_loss(
    apply_fn: Callable[..., Any],
    params: Dict[str, Any],
    batch: Float[Array, "b h w c"],
    alpha: float,
    beta: float,
    gamma: float,
) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
    x_rec, _, commit_loss, codebook_loss = apply_fn({"params": params}, batch)

    recon_loss = jnp.mean((x_rec - batch) ** 2)
    dy_true, dx_true = spatial_grads(batch)
    dy_rec, dx_rec = spatial_grads(x_rec)
    grad_loss = jnp.mean((dx_rec - dx_true) ** 2 + (dy_rec - dy_true) ** 2)
    w_true = compute_vorticity(batch)
    w_rec = compute_vorticity(x_rec)
    vorticity_loss = jnp.mean((w_rec - w_true) ** 2)

    total_loss = (
        recon_loss
        + alpha * grad_loss
        + gamma * vorticity_loss
        + codebook_loss
        + beta * commit_loss
    )

    metrics = {
        "loss": total_loss,
        "recon": recon_loss,
        "commit": commit_loss,
        "codebook": codebook_loss,
        "grad": grad_loss,
        "vorticity": vorticity_loss,
    }
    return total_loss, metrics


def tokenizer_st_loss(
    apply_fn: Callable[..., Any],
    params: Dict[str, Any],
    batch: Float[Array, "b t h w c"],
    alpha: float,
    beta: float,
    gamma: float,
    dropout_key: jax.Array,
) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
    outputs = apply_fn(
        {"params": params},
        {"videos": batch},
        training=True,
        rngs={"dropout": dropout_key},
    )
    # 1. recon loss and commit/codebook losses
    recon = outputs["recon"]
    recon_loss = jnp.mean((recon - batch) ** 2)
    q_loss = jnp.mean((jax.lax.stop_gradient(outputs["emb"]) - outputs["z"]) ** 2)
    commit_loss = jnp.mean((outputs["emb"] - jax.lax.stop_gradient(outputs["z"])) ** 2)
    # 2. physical losses
    b, t, h, w, c = batch.shape
    batch_flat = batch.reshape(b * t, h, w, c)
    recon_flat = recon.reshape(b * t, h, w, c)

    dy_true, dx_true = spatial_grads(batch_flat)
    dy_rec, dx_rec = spatial_grads(recon_flat)
    grad_loss = jnp.mean((dx_rec - dx_true) ** 2 + (dy_rec - dy_true) ** 2)

    w_true = compute_vorticity(batch_flat)
    w_rec = compute_vorticity(recon_flat)
    vorticity_loss = jnp.mean((w_rec - w_true) ** 2)

    # 3. sum up
    loss = (recon_loss + alpha * grad_loss + gamma * vorticity_loss
            + q_loss + beta * commit_loss)

    return loss, {
        "loss": loss,
        "recon": recon_loss,
        "q_loss": q_loss,
        "commit": commit_loss,
        "grad": grad_loss,
        "vorticity": vorticity_loss,
    }


def lam_loss(
    apply_fn: Callable[..., Any],
    params: Dict[str, Any],
    batch: Float[Array, "b t h w c"],
    beta: float,
    dropout_key: jax.Array,
) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
    outputs = apply_fn(
        {"params": params},
        {"videos": batch},
        training=True,
        rngs={"dropout": dropout_key},
    )
    gt_future = batch[:, 1:]
    recon = outputs["recon"]
    mse = jnp.mean((recon - gt_future) ** 2)
    q_loss = jnp.mean((jax.lax.stop_gradient(outputs["emb"]) - outputs["z"]) ** 2)
    commit_loss = jnp.mean((outputs["emb"] - jax.lax.stop_gradient(outputs["z"])) ** 2)
    loss = mse + q_loss + beta * commit_loss
    return loss, {
        "loss": loss,
        "mse": mse,
        "q_loss": q_loss,
        "commit": commit_loss,
    }


def dynamics_ar_loss(
    apply_fn: Callable[..., Any],
    params: Dict[str, Any],
    seq: Int[Array, "b l"],
    tok_tgt: Int[Array, "b l_out"],
    l_in: int,
    dropout_key: jax.Array,
    mask: Bool[Array, "b l_out"] | None = None,
    causal: bool = True,
) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
    logits = apply_fn(
        {"params": params},
        seq,
        train=True,
        rngs={"dropout": dropout_key},
        causal=causal,
    )
    logits_tgt = logits[:, l_in:, :]
    ce = optax.softmax_cross_entropy_with_integer_labels(logits_tgt, tok_tgt)
    if mask is not None:
        mask_f = mask.astype(ce.dtype)
        loss = (ce * mask_f).sum() / (mask_f.sum() + 1e-6)
    else:
        loss = ce.mean()
    return loss, {"loss": loss}


def dynamics_st_loss(
    apply_fn: Callable[..., Any],
    params: Dict[str, Any],
    tok_seq: Int[Array, "b t h w"],
    mask_key: jax.Array,
    dropout_key: jax.Array,
    latent_actions: Float[Array, "b t m d"] | None = None,
) -> Tuple[Float[Array, ""], Dict[str, Float[Array, ""]]]:
    batch = {"video_tokens": tok_seq, "mask_rng": mask_key}
    if latent_actions is not None:
        batch["latent_actions"] = latent_actions
    outputs = apply_fn(
        {"params": params},
        batch,
        training=True,
        rngs={"dropout": dropout_key},
    )
    mask = outputs["mask"].astype(jnp.float32)
    logits = outputs["token_logits"]

    b, t, n, v = logits.shape
    tok_seq_flat = tok_seq.reshape(b, t, n)

    ce = optax.softmax_cross_entropy_with_integer_labels(logits, tok_seq_flat)
    denom = jnp.maximum(mask.sum(), 1.0)
    loss = (mask * ce).sum() / denom
    acc = (mask * (logits.argmax(-1) == tok_seq_flat)).sum() / denom
    return loss, {"loss": loss, "masked_acc": acc}
