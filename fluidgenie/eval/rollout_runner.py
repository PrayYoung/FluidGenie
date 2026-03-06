from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
from jaxtyping import Array, Float, Int

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio  # type: ignore

from configs.eval_configs import RolloutConfig
from fluidgenie.eval.loaders import load_tokenizer, load_dynamics_model, load_lam_model
from fluidgenie.eval.metrics import compute_rollout_metrics
from fluidgenie.eval.samplers import (
    maskgit_rollout_tokens,
    st_maskgit_rollout_tokens,
    rollout_tokens_autoregressive,
    rollout_tokens_autoregressive_cached,
)
from fluidgenie.eval.utils import ensure_dir, resolve_stats_path, vq_decode_tokens, st_decode_tokens
from fluidgenie.models.lam import LatentActionModel
from configs.config_io import load_config_json


def _denorm(x: Float[Array, "... c"], mean, std, min_v=None, denom=None) -> Float[Array, "... c"]:
    if min_v is not None and denom is not None:
        return (x + 1.0) * 0.5 * denom + min_v
    if mean is None:
        return x
    return x * (std + 1e-6) + mean


def _norm(x: Float[Array, "... c"], mean, std, min_v=None, denom=None) -> Float[Array, "... c"]:
    if min_v is not None and denom is not None:
        return (x - min_v) / denom * 2.0 - 1.0
    if mean is None:
        return x
    return (x - mean) / (std + 1e-6)


def encode_context_tokens(
    vq_encode_tokens,
    tokenizer_params,
    ctx_frames: Float[Array, "t h w c"],
    mean,
    std,
    min_v=None,
    denom=None,
) -> Int[Array, "1 t h2 w2"]:
    # ctx_frames: [context,H,W,C] -> [1,context,h,w]
    x = _norm(ctx_frames.astype(np.float32), mean, std, min_v, denom)
    x_5d = jnp.array(x[None, ...], dtype=jnp.float32)
    tok = vq_encode_tokens(tokenizer_params, x_5d)
    return tok


def visualize_rollout(
    gt: Float[Array, "t h w c"],
    pred: Float[Array, "t h w c"],
    out_dir: Path,
    view: str,
    fps: int = 8,
) -> None:
    T, _, _, C = gt.shape
    frames = []
    for k in range(T):
        gt_k = gt[k]
        pr_k = pred[k]

        if view == "density" and C >= 3:
            w_gt = gt_k[..., 2]
            w_pr = pr_k[..., 2]
            err = np.abs(w_pr - w_gt)
            title = "Density"
        elif view == "speed" and C >= 2:
            w_gt = np.sqrt(gt_k[..., 0] ** 2 + gt_k[..., 1] ** 2)
            w_pr = np.sqrt(pr_k[..., 0] ** 2 + pr_k[..., 1] ** 2)
            err = np.abs(w_pr - w_gt)
            title = "Speed"
        elif view == "vorticity" and C >= 2:
            from fluidgenie.eval.utils import vorticity_from_uv
            w_gt = vorticity_from_uv(gt_k[..., :2])
            w_pr = vorticity_from_uv(pr_k[..., :2])
            err = np.abs(w_pr - w_gt)
            title = "Vorticity"
        else:
            w_gt = gt_k[..., 0]
            w_pr = pr_k[..., 0]
            err = np.abs(w_pr - w_gt)
            title = "Channel0"

        fig = plt.figure(figsize=(9, 3))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(w_gt)
        ax1.set_title(f"GT {title} (t={k})")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(w_pr)
        ax2.set_title(f"Rollout {title}")
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(err)
        ax3.set_title(f"Abs error {title}")
        ax3.axis("off")

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = np.array(buf)[..., :3]
        plt.close(fig)
        frames.append(img)

    out_gif = out_dir / f"gt_vs_rollout_{T}.gif"
    imageio.mimsave(out_gif, frames, duration=1.0 / max(1, fps))
    print("Saved:", out_gif)


def run_rollout(cfg: RolloutConfig) -> None:
    out = ensure_dir(Path(cfg.out_dir))
    if cfg.num_heads_tok is None:
        cfg.num_heads_tok = cfg.n_heads

    fields: Float[Array, "t h w c"] = np.load(cfg.npz_path, mmap_mode="r")
    T, H, W, C = fields.shape
    if cfg.start + cfg.context + cfg.horizon >= T:
        raise ValueError(
            f"Not enough frames in episode. Need start+context+horizon < T, got {cfg.start}+{cfg.context}+{cfg.horizon} >= {T}"
        )

    stats_path = resolve_stats_path(cfg.stats_path, cfg.vq_ckpt)
    if stats_path:
        stats = np.load(stats_path)
        if "min" in stats and "max" in stats:
            min_v = stats["min"].reshape(1, 1, 1, -1)
            max_v = stats["max"].reshape(1, 1, 1, -1)
            denom = (max_v - min_v) + 1e-6
            mean = None
            std = None
        else:
            mean = stats["mean"].reshape(1, 1, 1, -1)
            std = stats["std"].reshape(1, 1, 1, -1)
            min_v = None
            max_v = None
            denom = None
    else:
        mean = None
        std = None
        min_v = None
        max_v = None
        denom = None

    rng = jax.random.PRNGKey(cfg.seed)

    tok_model, tok_params, vq_encode_tokens, vq_cfg, codebook, dec_params = load_tokenizer(cfg, H, W, C)

    x0 = jnp.array(_norm(fields[cfg.start][None, ...], mean, std, min_v, denom), dtype=jnp.float32)

    x0_5d = x0[:,None, ...]  # [1,1,H,W,C]
    tok0 = vq_encode_tokens(tok_params, x0_5d)[:,0]
    h_tok, w_tok = int(tok0.shape[1]), int(tok0.shape[2])
    L_in = cfg.context * h_tok * w_tok
    L_out = h_tok * w_tok
    max_len = L_in + L_out
    print(f"[rollout] context={cfg.context} h_tok={h_tok} w_tok={w_tok} max_len={max_len}")
    dyn_cfg = load_config_json(cfg.dyn_ckpt)
    if dyn_cfg and isinstance(dyn_cfg.get("config"), dict):
        dyn_ctx = dyn_cfg["config"].get("context")
        dyn_model = dyn_cfg["config"].get("model")
        print(f"[rollout] dyn_ckpt: context={dyn_ctx} model={dyn_model}")
    assert max_len == (cfg.context + 1) * h_tok * w_tok, "max_len does not match context/token grid shape"

    dyn_model, dyn_params = load_dynamics_model(cfg, max_len=max_len, rng=rng)

    ctx_frames = fields[cfg.start : cfg.start + cfg.context]
    ctx_tok = encode_context_tokens(
        vq_encode_tokens,
        tok_params,
        ctx_frames,
        mean,
        std,
        min_v,
        denom,
    )

    lam_model, lam_params = load_lam_model(cfg, H, W, C, rng)
    ctx_frames_dyn = _norm(fields[cfg.start : cfg.start + cfg.context], mean, std, min_v, denom)

    def flatten_ctx(tok_ctx: Int[Array, "b t h w"]) -> Int[Array, "b l_in"]:
        return tok_ctx.reshape((tok_ctx.shape[0], -1))

    def unflatten_grid(tok_flat: Int[Array, "b l_out"]) -> Int[Array, "b h w"]:
        return tok_flat.reshape((tok_flat.shape[0], h_tok, w_tok))

    def rollout_step(carry, _):
        tok_ctx, step_rng = carry
        step_rng, rng = jax.random.split(step_rng)
        if cfg.model_type == "st_maskgit":
            latent_actions = None
            if cfg.use_lam:
                lam_out = lam_model.apply(
                    {"params": lam_params},
                    jnp.array(ctx_frames_dyn[None, ...], dtype=jnp.float32),
                    training=False,
                    method=LatentActionModel.vq_encode,
                )
                latent_actions = lam_out["z_q"]
                if latent_actions.shape[1] == cfg.context - 1:
                    pad = jnp.zeros(
                        (latent_actions.shape[0], 1, latent_actions.shape[2], latent_actions.shape[3]),
                        dtype=latent_actions.dtype,
                    )
                    latent_actions = jnp.concatenate([latent_actions, pad], axis=1)
            tok_next = st_maskgit_rollout_tokens(
                dyn_model,
                dyn_params,
                tok_ctx,
                mask_steps=cfg.mask_steps,
                rng_key=rng,
                init_mask_ratio=cfg.init_mask_ratio,
                latent_actions=latent_actions,
            )
        else:
            tok_in = flatten_ctx(tok_ctx).astype(jnp.int32)
            if cfg.model_type == "maskgit":
                tok_next_flat = maskgit_rollout_tokens(
                    dyn_model,
                    dyn_params,
                    tok_in,
                    L_out=L_out,
                    vocab=cfg.codebook_size + 1,
                    mask_token_id=cfg.codebook_size,
                    mask_steps=cfg.mask_steps,
                    rng_key=rng,
                )
            else:
                tok_next_flat = rollout_tokens_autoregressive(
                    dyn_model,
                    dyn_params,
                    tok_in,
                    L_out=L_out,
                    vocab=cfg.codebook_size,
                    bos_token_id=cfg.bos_token_id,
                )
            tok_next = unflatten_grid(tok_next_flat)

        # Maintain a context window of cfg.context frames.
        new_tok_ctx = jnp.concatenate([tok_ctx[:, 1:], tok_next[:, None, :, :]], axis=1)
        return (new_tok_ctx, step_rng), tok_next

    if cfg.model_type == "transformer" and cfg.use_kv_cache:
        def rollout_cached_step(tok_ctx, _):
            tok_in = flatten_ctx(tok_ctx).astype(jnp.int32)
            tok_next_flat = rollout_tokens_autoregressive_cached(
                dyn_model,
                dyn_params,
                tok_in,
                L_out=L_out,
                bos_token_id=cfg.bos_token_id,
                rng_seed=cfg.seed,
            )
            tok_next = unflatten_grid(tok_next_flat)
            new_tok_ctx = jnp.concatenate([tok_ctx[:, 1:], tok_next[:, None, :, :]], axis=1)
            return new_tok_ctx, tok_next

        _, tok_preds = jax.jit(lambda tc: jax.lax.scan(rollout_cached_step, tc, None, length=cfg.horizon))(ctx_tok)
    else:
        initial_carry = (ctx_tok, rng)
        _, tok_preds = jax.jit(lambda tc: jax.lax.scan(rollout_step, tc, None, length=cfg.horizon))(initial_carry)

    preds = []
    for k in range(cfg.horizon):
        tok_next = tok_preds[k]
        if cfg.tokenizer_arch == "st":
            x_hat = st_decode_tokens(tok_model, tok_params, tok_next, (H, W))
        else:
            x_hat = vq_decode_tokens(vq_cfg, dec_params, codebook, tok_next, out_channels=C)
        x_hat = _denorm(x_hat, mean, std, min_v, denom)
        preds.append(np.array(x_hat[0]))

    preds = np.stack(preds, axis=0)
    gt = fields[cfg.start + cfg.context : cfg.start + cfg.context + cfg.horizon]
    metrics = compute_rollout_metrics(gt, preds, view=cfg.view)

    np.savez_compressed(out / "rollout_arrays.npz", gt=gt, pred=preds, metrics=metrics)
    visualize_rollout(gt, preds, out, view=cfg.view)
