from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio  # type: ignore

from fluidgenie.models.vq_tokenizer import VQConfig
from fluidgenie.models.transformer_dynamics import DynConfig
from fluidgenie.eval.utils import (
    ensure_dir,
    vorticity_from_uv,
    load_vq_params,
    load_dyn_params,
    get_codebook_and_decoder_params,
    make_vq_encode_tokens,
    vq_decode_tokens,
)


def sample_argmax(logits: jnp.ndarray) -> jnp.ndarray:
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def rollout_tokens_autoregressive(dyn_model, dyn_params, tok_in: jnp.ndarray, L_out: int, vocab: int) -> jnp.ndarray:
    B, L_in = tok_in.shape
    tok_out = jnp.zeros((B, 0), dtype=jnp.int32)

    for i in range(L_out):
        bos = jnp.zeros((B, 1), dtype=jnp.int32)
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


def rollout_tokens_autoregressive_cached(dyn_model, dyn_params, tok_in: jnp.ndarray, L_out: int) -> jnp.ndarray:
    """
    Autoregressive rollout using KV cache (decode=True).
    """
    B, L_in = tok_in.shape
    rng = jax.random.PRNGKey(0)
    variables = dyn_model.init(rng, jnp.zeros((B, 1), dtype=jnp.int32), train=False, decode=True)
    cache = variables["cache"]

    # prefill cache with context tokens
    for i in range(L_in):
        token = tok_in[:, i:i+1]
        _, updated = dyn_model.apply({"params": dyn_params, "cache": cache}, token, train=False, decode=True, mutable=["cache"])
        cache = updated["cache"]

    # BOS token
    bos = jnp.zeros((B, 1), dtype=jnp.int32)
    logits, updated = dyn_model.apply({"params": dyn_params, "cache": cache}, bos, train=False, decode=True, mutable=["cache"])
    cache = updated["cache"]

    tok_out = []
    next_tok = sample_argmax(logits[:, -1, :])
    tok_out.append(next_tok)

    for _ in range(1, L_out):
        token = next_tok[:, None]
        logits, updated = dyn_model.apply({"params": dyn_params, "cache": cache}, token, train=False, decode=True, mutable=["cache"])
        cache = updated["cache"]
        next_tok = sample_argmax(logits[:, -1, :])
        tok_out.append(next_tok)

    tok_out = jnp.stack(tok_out, axis=1)
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
    tok_out = jnp.full((B, L_out), mask_token_id, dtype=jnp.int32)

    for step in range(mask_steps):
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
            idx = jnp.argsort(conf_row)[::-1]
            keep = idx[:k]
            tok_row = tok_row.at[keep].set(pred_row[keep])
            return tok_row

        tok_out = jax.vmap(update_one)(conf, pred, tok_out)

    return tok_out


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
) -> None:
    out = ensure_dir(Path(out_dir))

    data = np.load(npz_path, allow_pickle=True)
    fields = data["fields"]  # [T,H,W,C]
    T, H, W, C = fields.shape

    if start + context + horizon >= T:
        raise ValueError(f"Not enough frames in episode. Need start+context+horizon < T, got {start}+{context}+{horizon} >= {T}")

    vq_cfg = VQConfig(codebook_size=codebook_size, embed_dim=embed_dim, hidden=hidden)
    vq_model, vq_params = load_vq_params(vq_cfg, in_channels=C, H=H, W=W, ckpt_path=vq_ckpt)
    vq_encode_tokens = make_vq_encode_tokens(vq_model)

    if stats_path:
        stats = np.load(stats_path)
        mean = stats["mean"].reshape(1, 1, 1, -1)
        std = stats["std"].reshape(1, 1, 1, -1)
    else:
        mean = None
        std = None

    x0 = jnp.array(fields[start][None, ...], dtype=jnp.float32)
    if mean is not None:
        x0 = (x0 - mean) / (std + 1e-6)
    tok0 = vq_encode_tokens(vq_params, x0)  # [1,h,w]
    h_tok, w_tok = int(tok0.shape[1]), int(tok0.shape[2])
    L_in = context * h_tok * w_tok
    L_out = h_tok * w_tok
    max_len = L_in + L_out

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

    codebook, dec_params = get_codebook_and_decoder_params(vq_params)

    ctx_frames = fields[start : start + context]  # [context,H,W,C]
    ctx_tok_list = []
    for t in range(context):
        xt = jnp.array(ctx_frames[t][None, ...], dtype=jnp.float32)
        if mean is not None:
            xt = (xt - mean) / (std + 1e-6)
        ctx_tok_list.append(vq_encode_tokens(vq_params, xt)[0])  # [h,w]
    ctx_tok = jnp.stack(ctx_tok_list, axis=0)[None, ...]  # [1,context,h,w]

    def flatten_ctx(tok_ctx: jnp.ndarray) -> jnp.ndarray:
        return tok_ctx.reshape((tok_ctx.shape[0], -1))

    def unflatten_grid(tok_flat: jnp.ndarray) -> jnp.ndarray:
        return tok_flat.reshape((tok_flat.shape[0], h_tok, w_tok))

    preds = []
    gt = fields[start + context : start + context + horizon]  # [horizon,H,W,C]

    tok_ctx = ctx_tok
    for k in range(horizon):
        tok_in = flatten_ctx(tok_ctx).astype(jnp.int32)
        if model_type == "maskgit":
            tok_next_flat = maskgit_rollout_tokens(
                dyn_model,
                dyn_params,
                tok_in,
                L_out=L_out,
                vocab=vocab_size,
                mask_token_id=codebook_size,
                mask_steps=mask_steps,
            )
        else:
            if use_kv_cache:
                tok_next_flat = rollout_tokens_autoregressive_cached(dyn_model, dyn_params, tok_in, L_out=L_out)
            else:
                tok_next_flat = rollout_tokens_autoregressive(dyn_model, dyn_params, tok_in, L_out=L_out, vocab=codebook_size)
        tok_next = unflatten_grid(tok_next_flat)

        x_hat = vq_decode_tokens(vq_cfg, dec_params, codebook, tok_next, out_channels=C)
        if mean is not None:
            x_hat = x_hat * (std + 1e-6) + mean
        preds.append(np.array(x_hat[0]))

        tok_ctx = jnp.concatenate([tok_ctx[:, 1:], tok_next[:, None, :, :]], axis=1)

    preds = np.stack(preds, axis=0)

    frames = []
    for k in range(horizon):
        gt_k = gt[k]
        pr_k = preds[k]

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

    out_gif = out / f"gt_vs_rollout_{horizon}.gif"
    imageio.mimsave(out_gif, frames, duration=0.12)
    np.savez_compressed(out / "rollout_arrays.npz", gt=gt, pred=preds)

    print("Saved:", out_gif)
    print(f"Token grid: {h_tok}x{w_tok} | context={context} | horizon={horizon} | max_len={max_len}")
