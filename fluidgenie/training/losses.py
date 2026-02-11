from __future__ import annotations

from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp


def spatial_grads(img: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Central differences with edge padding to keep H,W sizes
    img_pad = jnp.pad(img, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    dx = 0.5 * (img_pad[:, 1:-1, 2:, :] - img_pad[:, 1:-1, :-2, :])
    dy = 0.5 * (img_pad[:, 2:, 1:-1, :] - img_pad[:, :-2, 1:-1, :])
    return dx, dy


def compute_vorticity(img: jnp.ndarray) -> jnp.ndarray:
    # img: [B,H,W,C], assumes C>=2 with u=0, v=1
    u = img[..., 0]
    v = img[..., 1]
    u_pad = jnp.pad(u, ((0, 0), (1, 1), (1, 1)), mode="edge")
    v_pad = jnp.pad(v, ((0, 0), (1, 1), (1, 1)), mode="edge")
    du_dy = 0.5 * (u_pad[:, 2:, 1:-1] - u_pad[:, :-2, 1:-1])
    dv_dx = 0.5 * (v_pad[:, 1:-1, 2:] - v_pad[:, 1:-1, :-2])
    return dv_dx - du_dy


def tokenizer_conv_loss(
    apply_fn,
    params: Dict[str, Any],
    batch: jnp.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
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
    apply_fn,
    params: Dict[str, Any],
    batch: jnp.ndarray,
    beta: float,
    dropout_key: jnp.ndarray,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    outputs = apply_fn(
        {"params": params},
        {"videos": batch},
        training=True,
        rngs={"dropout": dropout_key},
    )
    recon = outputs["recon"]
    mse = jnp.mean((recon - batch) ** 2)
    q_loss = jnp.mean((jax.lax.stop_gradient(outputs["emb"]) - outputs["z"]) ** 2)
    commit_loss = jnp.mean((outputs["emb"] - jax.lax.stop_gradient(outputs["z"])) ** 2)
    loss = mse + q_loss + beta * commit_loss
    return loss, {
        "loss": loss,
        "mse": mse,
        "q_loss": q_loss,
        "commit": commit_loss,
    }


def lam_loss(
    apply_fn,
    params: Dict[str, Any],
    batch: jnp.ndarray,
    beta: float,
    dropout_key: jnp.ndarray,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
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
