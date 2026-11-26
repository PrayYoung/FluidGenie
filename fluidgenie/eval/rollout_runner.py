from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from configs.eval_configs import RolloutConfig
from fluidgenie.eval.loaders import load_tokenizer, load_dynamics_model, load_lam_model
from fluidgenie.eval.metrics import compute_rollout_metrics
from fluidgenie.eval.samplers import (
    maskgit_rollout_tokens,
    st_maskgit_rollout_tokens,
    rollout_tokens_autoregressive,
    rollout_tokens_autoregressive_cached,
)
from fluidgenie.eval.utils import (
    ensure_dir,
    resolve_stats_path,
    vq_decode_tokens,
)
from fluidgenie.models.lam import LatentActionModel
from configs.config_io import load_config_json


def _denorm(
    x: Float[Array, "... c"], mean, std, min_v=None, denom=None
) -> Float[Array, "... c"]:
    if min_v is not None and denom is not None:
        return (x + 1.0) * 0.5 * denom + min_v
    if mean is None:
        return x
    return x * (std + 1e-6) + mean


def _norm(
    x: Float[Array, "... c"], mean, std, min_v=None, denom=None
) -> Float[Array, "... c"]:
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

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


def validate_lam_rollout(cfg):
    if cfg.lam_rollout_mode not in {"context", "oracle"}:
        raise ValueError("LAM rollout mode must be context or oracle")
    if cfg.use_lam:
        if cfg.context < 2:
            raise ValueError("LAM rollout needs context >= 2")
        if cfg.held_action_id < -1 or cfg.held_action_id >= cfg.lam_num_latents:
            raise ValueError("held_action_id exceeds the LAM codebook")
        if cfg.lam_rollout_mode == "oracle" and cfg.held_action_id >= 0:
            raise ValueError("Oracle and held action modes are mutually exclusive")


def run_rollout(cfg: RolloutConfig) -> None:
    validate_lam_rollout(cfg)
    if cfg.start < 0 or cfg.context < 1 or cfg.horizon < 1:
        raise ValueError("Invalid rollout bounds")
    dyn_cfg = load_config_json(cfg.dyn_ckpt)
    if not dyn_cfg:
        raise ValueError("Dynamics configuration is required")
    if not cfg.skip_split_check:
        from fluidgenie.data.splits import assert_checkpoint_evaluation

        split = assert_checkpoint_evaluation(
            dyn_cfg["config"], Path(cfg.npz_path).parent
        )
    else:
        split = {
            "disjoint": False,
            "status": "explicit synthetic counterfactual bypass; not generalization evidence",
        }
    out = ensure_dir(Path(cfg.out_dir))
    (out / "split_validation.json").write_text(json.dumps(split, indent=2))
    if cfg.num_heads_tok is None:
        cfg.num_heads_tok = cfg.n_heads

    fields: Float[Array, "t h w c"] = np.load(cfg.npz_path, mmap_mode="r")
    T, H, W, C = fields.shape
    if cfg.start + cfg.context + cfg.horizon > T:
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

    tok_model, tok_params, vq_encode_tokens, vq_cfg, codebook, dec_params = (
        load_tokenizer(cfg, H, W, C)
    )

    x0 = jnp.array(
        _norm(fields[cfg.start][None, ...], mean, std, min_v, denom), dtype=jnp.float32
    )

    x0_5d = x0[:, None, ...]  # [1,1,H,W,C]
    tok0 = vq_encode_tokens(tok_params, x0_5d)[:, 0]
    h_tok, w_tok = int(tok0.shape[1]), int(tok0.shape[2])
    L_in = cfg.context * h_tok * w_tok
    L_out = h_tok * w_tok
    max_len = L_in + L_out
    print(
        f"[rollout] context={cfg.context} h_tok={h_tok} w_tok={w_tok} max_len={max_len}"
    )
    dyn_cfg = load_config_json(cfg.dyn_ckpt)
    if dyn_cfg and isinstance(dyn_cfg.get("config"), dict):
        dyn_ctx = dyn_cfg["config"].get("context")
        dyn_model = dyn_cfg["config"].get("model")
        dyn_pred = int(dyn_cfg["config"].get("pred", 1))
        dyn_stride = int(dyn_cfg["config"].get("frame_stride", 1))
        print(
            f"[rollout] dyn_ckpt: context={dyn_ctx} model={dyn_model} "
            f"pred={dyn_pred} frame_stride={dyn_stride}"
        )
        if dyn_pred > 1 or dyn_stride > 1:
            print(
                "[rollout][warn] This rollout path predicts one frame per step. "
                "Multi-step dynamics checkpoints (pred>1 or frame_stride>1) "
                "need a dedicated multi-step inference path. TODO: add multi-step rollout."
            )
    assert max_len == (cfg.context + 1) * h_tok * w_tok, (
        "max_len does not match context/token grid shape"
    )

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
    ctx_frames_dyn = _norm(
        fields[cfg.start : cfg.start + cfg.context], mean, std, min_v, denom
    )

    def flatten_ctx(tok_ctx: Int[Array, "b t h w"]) -> Int[Array, "b l_in"]:
        return tok_ctx.reshape((tok_ctx.shape[0], -1))

    def unflatten_grid(tok_flat: Int[Array, "b l_out"]) -> Int[Array, "b h w"]:
        return tok_flat.reshape((tok_flat.shape[0], h_tok, w_tok))

    preds = []
    normalized_context = jnp.asarray(ctx_frames_dyn[None], dtype=jnp.float32)
    for k in range(cfg.horizon):
        rng, step_rng = jax.random.split(rng)
        if cfg.model_type == "st_maskgit":
            latent_actions = None
            if cfg.use_lam:
                observed = normalized_context
                if cfg.lam_rollout_mode == "oracle":
                    end = cfg.start + cfg.context + k + 1
                    observed = jnp.asarray(
                        _norm(
                            fields[end - cfg.context - 1 : end], mean, std, min_v, denom
                        )[None],
                        dtype=jnp.float32,
                    )
                encoded = lam_model.apply(
                    {"params": lam_params},
                    observed,
                    training=False,
                    method=LatentActionModel.vq_encode,
                )["z_q"]
                latent_actions = encoded[:, -(cfg.context - 1) :]
                if cfg.held_action_id >= 0:
                    if cfg.held_action_id >= cfg.lam_num_latents:
                        raise ValueError("held_action_id exceeds the LAM codebook")
                    chosen = lam_model.apply(
                        {"params": lam_params},
                        jnp.asarray([cfg.held_action_id]),
                        method=lambda module, ids: module.vq.get_codes(ids),
                    )
                    next_action = chosen[:, None, None, :]
                else:
                    # Reuse a real inferred code for the next transition, never a zero vector.
                    next_action = encoded[:, -1:]
                if cfg.lam_rollout_mode == "oracle":
                    latent_actions = encoded[:, -cfg.context :]
                else:
                    latent_actions = jnp.concatenate(
                        [latent_actions, next_action], axis=1
                    )
            tok_next = st_maskgit_rollout_tokens(
                dyn_model,
                dyn_params,
                ctx_tok,
                mask_steps=cfg.mask_steps,
                rng_key=step_rng,
                init_mask_ratio=cfg.init_mask_ratio,
                latent_actions=latent_actions,
            )
        else:
            tok_in = flatten_ctx(ctx_tok).astype(jnp.int32)
            if cfg.model_type == "maskgit":
                tok_flat = maskgit_rollout_tokens(
                    dyn_model,
                    dyn_params,
                    tok_in,
                    L_out=L_out,
                    vocab=cfg.codebook_size + 1,
                    mask_token_id=cfg.codebook_size,
                    mask_steps=cfg.mask_steps,
                    rng_key=step_rng,
                )
            elif cfg.use_kv_cache:
                tok_flat = rollout_tokens_autoregressive_cached(
                    dyn_model,
                    dyn_params,
                    tok_in,
                    L_out=L_out,
                    bos_token_id=cfg.bos_token_id,
                    rng_seed=cfg.seed + k,
                )
            else:
                tok_flat = rollout_tokens_autoregressive(
                    dyn_model,
                    dyn_params,
                    tok_in,
                    L_out=L_out,
                    vocab=cfg.codebook_size,
                    bos_token_id=cfg.bos_token_id,
                )
            tok_next = unflatten_grid(tok_flat)
        joint_tokens = jnp.concatenate([ctx_tok, tok_next[:, None]], axis=1)
        if cfg.tokenizer_arch == "st":
            decoded = tok_model.apply(
                {"params": tok_params},
                joint_tokens,
                (H, W),
                method=tok_model.decode_video_tokens,
            )
            next_normalized = decoded[:, -1]
        else:
            next_normalized = vq_decode_tokens(
                vq_cfg, dec_params, codebook, tok_next, out_channels=C
            )
        preds.append(np.asarray(_denorm(next_normalized, mean, std, min_v, denom)[0]))
        normalized_context = jnp.concatenate(
            [normalized_context[:, 1:], next_normalized[:, None]], axis=1
        )
        ctx_tok = joint_tokens[:, 1:]
    preds = np.stack(preds, axis=0)

    (out / "rollout_protocol.json").write_text(
        json.dumps(
            {
                "lam_mode": cfg.lam_rollout_mode,
                "oracle_uses_future": bool(
                    cfg.use_lam and cfg.lam_rollout_mode == "oracle"
                ),
                "held_action_id": cfg.held_action_id,
                "joint_causal_st_decode": cfg.tokenizer_arch == "st",
            },
            indent=2,
        )
    )
    gt = fields[cfg.start + cfg.context : cfg.start + cfg.context + cfg.horizon]
    metrics = compute_rollout_metrics(gt, preds, view=cfg.view)

    np.savez_compressed(out / "rollout_arrays.npz", gt=gt, pred=preds, metrics=metrics)
    visualize_rollout(gt, preds, out, view=cfg.view)
