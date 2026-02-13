from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flax.core import freeze, unfreeze

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio  # type: ignore

from fluidgenie.models.vq_tokenizer import VQConfig
from fluidgenie.models.transformer_dynamics import DynConfig
from fluidgenie.models.dynamics_st import DynamicsSTMaskGIT
from fluidgenie.models.lam import LatentActionModel
from fluidgenie.eval.utils import (
    ensure_dir,
    vorticity_from_uv,
    load_tokenizer_params,
    load_dyn_params,
    get_codebook_and_decoder_params,
    make_vq_encode_tokens,
    make_st_encode_tokens,
    vq_decode_tokens,
    st_decode_tokens,
)
from fluidgenie.training.checkpoint_utils import load_params
from tqdm import tqdm


def sample_argmax(logits: jnp.ndarray) -> jnp.ndarray:
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def _denorm(x: jnp.ndarray, mean, std) -> jnp.ndarray:
    if mean is None:
        return x
    return x * (std + 1e-6) + mean


def _norm(x: np.ndarray, mean, std) -> np.ndarray:
    if mean is None:
        return x
    return (x - mean) / (std + 1e-6)


def compute_rollout_metrics(gt: np.ndarray, pred: np.ndarray, view: str) -> dict:
    """
    gt, pred: [T,H,W,C]
    Returns per-frame metrics and aggregates.
    """
    assert gt.shape == pred.shape
    T, _, _, C = gt.shape
    mse = np.mean((gt - pred) ** 2, axis=(1, 2, 3))
    mae = np.mean(np.abs(gt - pred), axis=(1, 2, 3))

    if view == "density" and C >= 3:
        gt_v = gt[..., 2]
        pr_v = pred[..., 2]
    elif view == "speed" and C >= 2:
        gt_v = np.sqrt(gt[..., 0] ** 2 + gt[..., 1] ** 2)
        pr_v = np.sqrt(pred[..., 0] ** 2 + pred[..., 1] ** 2)
    elif view == "vorticity" and C >= 2:
        gt_v = np.stack([vorticity_from_uv(gt[t, ..., :2]) for t in range(T)], axis=0)
        pr_v = np.stack([vorticity_from_uv(pred[t, ..., :2]) for t in range(T)], axis=0)
    else:
        gt_v = gt[..., 0]
        pr_v = pred[..., 0]

    view_mae = np.mean(np.abs(pr_v - gt_v), axis=(1, 2))

    return {
        "mse": mse,
        "mae": mae,
        "view_mae": view_mae,
        "mse_mean": float(mse.mean()),
        "mae_mean": float(mae.mean()),
        "view_mae_mean": float(view_mae.mean()),
    }


def visualize_rollout(gt: np.ndarray, pred: np.ndarray, out_dir: Path, view: str, fps: int = 8) -> None:
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


def encode_context_tokens(vq_encode_tokens, vq_params, ctx_frames: np.ndarray, mean, std) -> jnp.ndarray:
    # ctx_frames: [context,H,W,C] -> [1,context,h,w]
    x = ctx_frames.astype(np.float32)
    if mean is not None:
        x = (x - mean) / (std + 1e-6)
    x = jnp.array(x, dtype=jnp.float32)
    # vmap over time dimension
    tok = jax.vmap(lambda f: vq_encode_tokens(vq_params, f[None, ...])[0])(x)
    return tok[None, ...]


def rollout_tokens_autoregressive(
    dyn_model,
    dyn_params,
    tok_in: jnp.ndarray,
    L_out: int,
    vocab: int,
    bos_token_id: int = 0,
) -> jnp.ndarray:
    B, L_in = tok_in.shape
    tok_out = jnp.zeros((B, 0), dtype=jnp.int32)

    for i in range(L_out):
        bos = jnp.full((B, 1), bos_token_id, dtype=jnp.int32)
        tok_tgt_in = jnp.concatenate([bos, tok_out], axis=1)  # [B, 1+i]

        need = L_out - tok_tgt_in.shape[1]
        if need > 0:
            tok_tgt_in = jnp.concatenate([tok_tgt_in, jnp.zeros((B, need), dtype=jnp.int32)], axis=1)

        seq = jnp.concatenate([tok_in, tok_tgt_in], axis=1)  # [B, L_in+L_out]

        logits = dyn_model.apply({"params": dyn_params}, seq, train=False)  # [B, L, vocab]
        log_i = logits[:, L_in + i, :]  # [B, vocab]
        next_tok = sample_argmax(log_i)[:, None]  # [B,1]
        tok_out = jnp.concatenate([tok_out, next_tok], axis=1)

    return tok_out


def rollout_tokens_autoregressive_cached(
    dyn_model,
    dyn_params,
    tok_in: jnp.ndarray,
    L_out: int,
    bos_token_id: int = 0,
    rng_seed: int = 0,
) -> jnp.ndarray:
    """
    Autoregressive rollout using KV cache (decode=True).
    """
    B, L_in = tok_in.shape
    rng = jax.random.PRNGKey(rng_seed)
    variables = dyn_model.init(rng, jnp.zeros((B, 1), dtype=jnp.int32), train=False, decode=True)
    cache = variables["cache"]
    # reset cache index to 0 because init() advances it by 1
    cache_mut = unfreeze(cache)
    cache_mut["cache_index"] = jnp.array(0)
    cache = freeze(cache_mut)

    # prefill cache with context tokens (scan for compilation)
    def prefill_step(carry_cache, token_slice):
        token = token_slice[:, None]
        _, updated = dyn_model.apply(
            {"params": dyn_params, "cache": carry_cache},
            token,
            train=False,
            decode=True,
            mutable=["cache"],
        )
        return updated["cache"], None

    cache, _ = jax.lax.scan(prefill_step, cache, tok_in.T)

    # BOS token
    bos = jnp.full((B, 1), bos_token_id, dtype=jnp.int32)
    logits, updated = dyn_model.apply(
        {"params": dyn_params, "cache": cache},
        bos,
        train=False,
        decode=True,
        mutable=["cache"],
    )
    cache = updated["cache"]

    next_tok = sample_argmax(logits[:, -1, :])

    def gen_step(carry, _):
        carry_cache, prev_tok = carry
        token = prev_tok[:, None]
        logits, updated = dyn_model.apply(
            {"params": dyn_params, "cache": carry_cache},
            token,
            train=False,
            decode=True,
            mutable=["cache"],
        )
        next_t = sample_argmax(logits[:, -1, :])
        return (updated["cache"], next_t), next_t

    if L_out <= 1:
        return next_tok[:, None]

    (_, _), tok_rest = jax.lax.scan(gen_step, (cache, next_tok), None, length=L_out - 1)
    tok_out = jnp.concatenate([next_tok[:, None], tok_rest.T], axis=1)
    return tok_out


def maskgit_rollout_tokens(
    dyn_model,
    dyn_params,
    tok_in: jnp.ndarray,
    L_out: int,
    vocab: int,
    mask_token_id: int,
    mask_steps: int,
) -> jnp.ndarray:
    B, L_in = tok_in.shape

    def step_fn(tok_out, step):
        seq = jnp.concatenate([tok_in, tok_out], axis=1)
        logits = dyn_model.apply({"params": dyn_params}, seq, train=False)
        logits_tgt = logits[:, L_in:, :]
        probs = jax.nn.softmax(logits_tgt, axis=-1)
        pred = jnp.argmax(probs, axis=-1).astype(jnp.int32)
        conf = jnp.max(probs, axis=-1)

        # mask schedule (cosine)
        t = (step + 1) / mask_steps
        ratio = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
        k = jnp.maximum(1, jnp.floor(ratio * L_out).astype(jnp.int32))

        def update_one(conf_row, pred_row, tok_row):
            _, keep = jax.lax.top_k(conf_row, k)
            tok_row = tok_row.at[keep].set(pred_row[keep])
            return tok_row

        tok_out = jax.vmap(update_one)(conf, pred, tok_out)
        return tok_out, None

    tok_init = jnp.full((B, L_out), mask_token_id, dtype=jnp.int32)
    tok_out, _ = jax.lax.scan(step_fn, tok_init, jnp.arange(mask_steps))
    return tok_out


def st_maskgit_rollout_tokens(
    dyn_model,
    dyn_params,
    tok_ctx: jnp.ndarray,
    mask_steps: int,
    rng_key: jnp.ndarray,
    latent_actions: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    tok_ctx: [B, context, h, w]
    returns tok_next: [B, h, w]
    """
    b, _, h, w = tok_ctx.shape
    n = h * w
    tok_next0 = jnp.zeros((b, h, w), dtype=jnp.int32)
    mask0 = jnp.ones((b, h, w), dtype=jnp.bool_)

    def step_fn(carry, step_idx):
        tok_next, mask, rng = carry
        tok_seq = jnp.concatenate([tok_ctx, tok_next[:, None]], axis=1)  # [B, T+1, h, w]
        step_key, rng = jax.random.split(rng)
        batch = {
            "video_tokens": tok_seq,
            "mask_rng": step_key,
            "mask": jnp.concatenate([jnp.zeros_like(tok_ctx, dtype=jnp.bool_), mask[:, None]], axis=1),
        }
        if latent_actions is not None:
            batch["latent_actions"] = latent_actions
        logits = dyn_model.apply({"params": dyn_params}, batch, training=False)
        logits_last = logits["token_logits"][:, -1]  # [B, N, vocab]
        probs = jax.nn.softmax(logits_last, axis=-1)
        pred = jnp.argmax(probs, axis=-1).astype(jnp.int32)
        conf = jnp.max(probs, axis=-1)

        t_ratio = (step_idx + 1) / mask_steps
        ratio = 0.5 * (1.0 + jnp.cos(jnp.pi * t_ratio))
        k = jnp.maximum(1, jnp.floor(ratio * n).astype(jnp.int32))

        def update_one(conf_row, pred_row, tok_row):
            _, keep = jax.lax.top_k(conf_row, k)
            tok_row = tok_row.at[keep].set(pred_row[keep])
            return tok_row, keep

        tok_next_flat = tok_next.reshape(b, n)
        tok_next_flat, keep_idx = jax.vmap(update_one)(conf, pred, tok_next_flat)
        tok_next = tok_next_flat.reshape(b, h, w)

        def update_mask(keep):
            m = jnp.ones((n,), dtype=jnp.bool_)
            m = m.at[keep].set(False)
            return m

        mask_flat = jax.vmap(update_mask)(keep_idx)
        mask = mask_flat.reshape(b, h, w)
        return (tok_next, mask, rng), None

    (tok_next, _, _), _ = jax.lax.scan(step_fn, (tok_next0, mask0, rng_key), jnp.arange(mask_steps))
    return tok_next


def run_rollout_generator(
    npz_path: str,
    vq_ckpt: str,
    dyn_ckpt: str,
    start: int,
    horizon: int,
    context: int,
    codebook_size: int,
    embed_dim: int,
    hidden: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    model_type: str = "transformer",
    use_kv_cache: bool = True,
    mask_steps: int = 8,
    stats_path: Optional[str] = None,
    tokenizer_arch: str = "conv",
    patch_size: int = 4,
    model_dim: int = 256,
    num_blocks: int = 6,
    num_heads_tok: int = 8,
    tokenizer_dropout: float = 0.0,
    codebook_dropout: float = 0.0,
    use_lam: bool = False,
    lam_ckpt: str = "",
    lam_model_dim: int = 256,
    lam_latent_dim: int = 64,
    lam_num_latents: int = 128,
    lam_patch_size: int = 4,
    lam_num_blocks: int = 6,
    lam_num_heads: int = 8,
    lam_dropout: float = 0.0,
    lam_codebook_dropout: float = 0.0,
    bos_token_id: int = 0,
    rng_seed: int = 0,
    view: str = "density",
) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(npz_path, allow_pickle=True)
    fields = data["fields"]  # [T,H,W,C]
    T, H, W, C = fields.shape

    if start + context + horizon >= T:
        raise ValueError(f"Not enough frames in episode. Need start+context+horizon < T, got {start}+{context}+{horizon} >= {T}")

    vq_cfg = VQConfig(codebook_size=codebook_size, embed_dim=embed_dim, hidden=hidden)
    vq_model, vq_params = load_tokenizer_params(
        tokenizer_arch,
        vq_cfg,
        in_channels=C,
        H=H,
        W=W,
        ckpt_path=vq_ckpt,
        patch_size=patch_size,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads_tok,
        dropout=tokenizer_dropout,
        codebook_dropout=codebook_dropout,
    )
    if tokenizer_arch == "st":
        vq_encode_tokens = make_st_encode_tokens(vq_model)
    else:
        vq_encode_tokens = make_vq_encode_tokens(vq_model)

    if stats_path:
        stats = np.load(stats_path)
        mean = stats["mean"].reshape(1, 1, 1, -1)
        std = stats["std"].reshape(1, 1, 1, -1)
    else:
        mean = None
        std = None

    x0 = jnp.array(_norm(fields[start][None, ...], mean, std), dtype=jnp.float32)
    tok0 = vq_encode_tokens(vq_params, x0)
    h_tok, w_tok = int(tok0.shape[1]), int(tok0.shape[2])
    L_in = context * h_tok * w_tok
    L_out = h_tok * w_tok
    max_len = L_in + L_out

    rng = jax.random.PRNGKey(rng_seed)
    if model_type == "st_maskgit":
        rng, mask_rng = jax.random.split(rng)
        dyn_model = DynamicsSTMaskGIT(
            model_dim=d_model,
            num_latents=codebook_size,
            num_blocks=n_layers,
            num_heads=n_heads,
            dropout=dropout,
            mask_ratio_min=0.0,
            mask_ratio_max=1.0,
        )
        tok_seq0 = jnp.concatenate(
            [jnp.repeat(tok0[:, None, ...], context, axis=1), tok0[:, None, ...]], axis=1
        )
        dyn_init = dyn_model.init(rng, {"video_tokens": tok_seq0, "mask_rng": mask_rng}, training=False)["params"]
        dyn_params = load_params(dyn_ckpt, dyn_init)
    else:
        vocab_size = codebook_size + (1 if model_type == "maskgit" else 0)
        dyn_cfg = DynConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len,
        )
        dyn_model, dyn_params = load_dyn_params(dyn_cfg, max_len=max_len, ckpt_path=dyn_ckpt)

    if tokenizer_arch == "conv":
        codebook, dec_params = get_codebook_and_decoder_params(vq_params)
    else:
        codebook = None
        dec_params = None

    ctx_frames = fields[start : start + context]
    ctx_tok = encode_context_tokens(vq_encode_tokens, vq_params, ctx_frames, mean, std)

    lam_model = None
    lam_params = None
    if use_lam:
        rng, lam_rng = jax.random.split(rng)
        if not lam_ckpt:
            raise ValueError("--lam_ckpt is required when use_lam=True")
        lam_model = LatentActionModel(
            in_dim=C,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            num_latents=lam_num_latents,
            patch_size=lam_patch_size,
            num_blocks=lam_num_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout,
            codebook_dropout=lam_codebook_dropout,
        )
        lam_init = lam_model.init(
            lam_rng,
            {"videos": jnp.zeros((1, context, H, W, C), dtype=jnp.float32)},
            training=False,
        )["params"]
        lam_params = load_params(lam_ckpt, lam_init)

    ctx_frames_dyn = _norm(fields[start : start + context], mean, std)

    def flatten_ctx(tok_ctx: jnp.ndarray) -> jnp.ndarray:
        return tok_ctx.reshape((tok_ctx.shape[0], -1))

    def unflatten_grid(tok_flat: jnp.ndarray) -> jnp.ndarray:
        return tok_flat.reshape((tok_flat.shape[0], h_tok, w_tok))

    def rollout_step(tok_ctx, _):
        tok_in = flatten_ctx(tok_ctx).astype(jnp.int32)
        if model_type == "st_maskgit":
            latent_actions = None
            if use_lam:
                lam_out = lam_model.apply(
                    {"params": lam_params},
                    jnp.array(ctx_frames_dyn[None, ...], dtype=jnp.float32),
                    training=False,
                    method=LatentActionModel.vq_encode,
                )
                latent_actions = lam_out["z_q"]
            tok_next = st_maskgit_rollout_tokens(
                dyn_model,
                dyn_params,
                tok_ctx,
                mask_steps=mask_steps,
                rng_key=rng,
                latent_actions=latent_actions,
            )
        else:
            if model_type == "maskgit":
                tok_next_flat = maskgit_rollout_tokens(
                    dyn_model,
                    dyn_params,
                    tok_in,
                    L_out=L_out,
                    vocab=codebook_size + 1,
                    mask_token_id=codebook_size,
                    mask_steps=mask_steps,
                )
            else:
                tok_next_flat = rollout_tokens_autoregressive(
                    dyn_model,
                    dyn_params,
                    tok_in,
                    L_out=L_out,
                    vocab=codebook_size,
                    bos_token_id=bos_token_id,
                )
            tok_next = unflatten_grid(tok_next_flat)

        new_tok_ctx = jnp.concatenate([tok_ctx[:, 1:], tok_next[:, None, :, :]], axis=1)
        return new_tok_ctx, tok_next

    if model_type == "transformer" and use_kv_cache:
        def rollout_cached_step(tok_ctx, _):
            tok_in = flatten_ctx(tok_ctx).astype(jnp.int32)
            tok_next_flat = rollout_tokens_autoregressive_cached(
                dyn_model,
                dyn_params,
                tok_in,
                L_out=L_out,
                bos_token_id=bos_token_id,
                rng_seed=rng_seed,
            )
            tok_next = unflatten_grid(tok_next_flat)
            new_tok_ctx = jnp.concatenate([tok_ctx[:, 1:], tok_next[:, None, :, :]], axis=1)
            return new_tok_ctx, tok_next

        _, tok_preds = jax.jit(lambda tc: jax.lax.scan(rollout_cached_step, tc, None, length=horizon))(ctx_tok)
    else:
        _, tok_preds = jax.jit(lambda tc: jax.lax.scan(rollout_step, tc, None, length=horizon))(ctx_tok)

    preds = []
    for k in tqdm(range(horizon), desc="decode"):
        tok_next = tok_preds[k][None, ...]
        if tokenizer_arch == "st":
            x_hat = st_decode_tokens(vq_model, vq_params, tok_next, (H, W))
        else:
            x_hat = vq_decode_tokens(vq_cfg, dec_params, codebook, tok_next, out_channels=C)
        x_hat = _denorm(x_hat, mean, std)
        preds.append(np.array(x_hat[0]))

    preds = np.stack(preds, axis=0)
    gt = fields[start + context : start + context + horizon]
    metrics = compute_rollout_metrics(gt, preds, view=view)
    return gt, preds, metrics


def run_rollout(
    npz_path: str,
    vq_ckpt: str,
    dyn_ckpt: str,
    out_dir: str,
    start: int,
    horizon: int,
    context: int,
    codebook_size: int,
    embed_dim: int,
    hidden: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    model_type: str = "transformer",
    use_kv_cache: bool = True,
    mask_steps: int = 8,
    view: str = "density",
    stats_path: Optional[str] = None,
    tokenizer_arch: str = "conv",
    patch_size: int = 4,
    model_dim: int = 256,
    num_blocks: int = 6,
    num_heads: int = 8,
    num_heads_tok: Optional[int] = None,
    tokenizer_dropout: float = 0.0,
    codebook_dropout: float = 0.0,
    use_lam: bool = False,
    lam_ckpt: str = "",
    lam_model_dim: int = 256,
    lam_latent_dim: int = 64,
    lam_num_latents: int = 128,
    lam_patch_size: int = 4,
    lam_num_blocks: int = 6,
    lam_num_heads: int = 8,
    lam_dropout: float = 0.0,
    lam_codebook_dropout: float = 0.0,
    bos_token_id: int = 0,
    rng_seed: int = 0,
) -> None:
    out = ensure_dir(Path(out_dir))
    if num_heads_tok is None:
        num_heads_tok = num_heads

    gt, preds, metrics = run_rollout_generator(
        npz_path=npz_path,
        vq_ckpt=vq_ckpt,
        dyn_ckpt=dyn_ckpt,
        start=start,
        horizon=horizon,
        context=context,
        codebook_size=codebook_size,
        embed_dim=embed_dim,
        hidden=hidden,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        model_type=model_type,
        use_kv_cache=use_kv_cache,
        mask_steps=mask_steps,
        stats_path=stats_path,
        tokenizer_arch=tokenizer_arch,
        patch_size=patch_size,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads_tok=num_heads_tok,
        tokenizer_dropout=tokenizer_dropout,
        codebook_dropout=codebook_dropout,
        use_lam=use_lam,
        lam_ckpt=lam_ckpt,
        lam_model_dim=lam_model_dim,
        lam_latent_dim=lam_latent_dim,
        lam_num_latents=lam_num_latents,
        lam_patch_size=lam_patch_size,
        lam_num_blocks=lam_num_blocks,
        lam_num_heads=lam_num_heads,
        lam_dropout=lam_dropout,
        lam_codebook_dropout=lam_codebook_dropout,
        bos_token_id=bos_token_id,
        rng_seed=rng_seed,
        view=view,
    )

    np.savez_compressed(out / "rollout_arrays.npz", gt=gt, pred=preds, metrics=metrics)
    visualize_rollout(gt, preds, out, view=view)
