"""
Unified demo entry for FluidGenie.

Modes:
  1) tokenizer: VQ recon + token map (good for README)
  2) rollout:   token-space autoregressive rollout -> decode -> GIF

Examples:

# 1) Tokenizer recon (requires VQ ckpt)
uv run python -m fluidgenie.cli.demo \
  --mode tokenizer \
  --npz data/ns2d/episode_000000.npz \
  --vq_ckpt runs/vq/latest.ckpt \
  --out demo/tokenizer \
  --frame 0 \
  --codebook 512 --embed 64 --hidden 128

# 2) Rollout GIF (requires VQ ckpt + Dynamics ckpt)
uv run python -m fluidgenie.cli.demo \
  --mode rollout \
  --npz data/ns2d/episode_000000.npz \
  --vq_ckpt runs/vq/latest.ckpt \
  --dyn_ckpt runs/dyn/latest.ckpt \
  --out demo/rollout \
  --start 0 --horizon 60 --context 2 \
  --codebook 512 --embed 64 --hidden 128 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

import jax
import jax.numpy as jnp
from flax.serialization import from_bytes

import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio  # type: ignore

from fluidgenie.models.vq_tokenizer import VQVAE, VQConfig, Decoder
from fluidgenie.models.transformer_dynamics import TransformerDynamics, DynConfig


# ----------------------------
# Utils: physics-ish viz
# ----------------------------

def vorticity_from_uv(uv: np.ndarray) -> np.ndarray:
    """uv: [H,W,2] -> vorticity [H,W] using finite differences."""
    u = uv[..., 0]
    v = uv[..., 1]
    dudY = np.gradient(u, axis=0)
    dvdX = np.gradient(v, axis=1)
    return dvdX - dudY


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------
# Utils: load ckpts
# ----------------------------

def load_vq_params(vq_cfg: VQConfig, in_channels: int, H: int, W: int, ckpt_path: str, seed: int = 0):
    model = VQVAE(vq_cfg, in_channels=in_channels)
    rng = jax.random.PRNGKey(seed)
    params_init = model.init(rng, jnp.zeros((1, H, W, in_channels), dtype=jnp.float32))["params"]
    params = from_bytes(params_init, Path(ckpt_path).read_bytes())
    return model, params


def load_dyn_params(dyn_cfg: DynConfig, max_len: int, ckpt_path: str, seed: int = 0):
    model = TransformerDynamics(dyn_cfg)
    rng = jax.random.PRNGKey(seed)
    params_init = model.init(rng, jnp.zeros((1, max_len), dtype=jnp.int32), train=False)["params"]
    params = from_bytes(params_init, Path(ckpt_path).read_bytes())
    return model, params


def get_codebook_and_decoder_params(vq_params: dict) -> Tuple[jnp.ndarray, dict]:
    """
    VQVAE params are typically:
      Encoder_0, VectorQuantizer_0(codebook), Decoder_0
    We keep a fallback search in case module naming differs.
    """
    # common case
    if "VectorQuantizer_0" in vq_params and "Decoder_0" in vq_params:
        codebook = vq_params["VectorQuantizer_0"]["codebook"]
        dec_params = vq_params["Decoder_0"]
        return codebook, dec_params

    # fallback: search for "codebook"
    codebook = None
    dec_params = None
    for k, v in vq_params.items():
        if isinstance(v, dict) and "codebook" in v:
            codebook = v["codebook"]
        if k.lower().startswith("decoder") and isinstance(v, dict):
            dec_params = v
    if codebook is None or dec_params is None:
        raise KeyError("Could not locate codebook/decoder params inside VQ params dict.")
    return codebook, dec_params


# ----------------------------
# Tokenization / detokenization
# ----------------------------

def make_vq_encode_tokens(vq_model: VQVAE):
    @jax.jit
    def _encode(vq_params: dict, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [B,H,W,C] -> tok: [B,h,w] int32
        """
        _x_rec, tok, _commit, _cb = vq_model.apply({"params": vq_params}, x)
        return tok.astype(jnp.int32)
    return _encode


@jax.jit
def vq_decode_tokens(vq_cfg: VQConfig, dec_params: dict, codebook: jnp.ndarray, tok: jnp.ndarray, out_channels: int) -> jnp.ndarray:
    """
    tok: [B,h,w] -> x_hat: [B,H,W,C]
    """
    # tok -> z_q
    z_q = codebook[tok]  # [B,h,w,D]
    decoder = Decoder(vq_cfg, out_channels=out_channels)
    x_hat = decoder.apply({"params": dec_params}, z_q)
    return x_hat


# ----------------------------
# Demo: tokenizer recon
# ----------------------------

def demo_tokenizer(npz_path: str, vq_ckpt: str, out_dir: str, frame: int,
                   codebook_size: int, embed_dim: int, hidden: int, stats_path: str | None = None):
    out = ensure_dir(Path(out_dir))

    data = np.load(npz_path, allow_pickle=True)
    fields = data["fields"]  # [T,H,W,C]
    x = fields[frame]        # [H,W,C]
    H, W, C = x.shape

    vq_cfg = VQConfig(codebook_size=codebook_size, embed_dim=embed_dim, hidden=hidden)
    vq_model, vq_params = load_vq_params(vq_cfg, in_channels=C, H=H, W=W, ckpt_path=vq_ckpt)

    if stats_path:
        stats = np.load(stats_path)
        mean = stats["mean"].reshape(1, 1, -1)
        std = stats["std"].reshape(1, 1, -1)
        x_norm = (x - mean) / (std + 1e-6)
    else:
        x_norm = x

    x_in = jnp.array(x_norm[None, ...], dtype=jnp.float32)
    x_rec, tok, commit, cb = vq_model.apply({"params": vq_params}, x_in)
    x_rec = np.array(x_rec[0])
    if stats_path:
        x_rec = x_rec * (std + 1e-6) + mean
    tok = np.array(tok[0])

    fig = plt.figure(figsize=(12, 4))

    if C >= 2:
        w_gt = vorticity_from_uv(x[..., :2])
        w_rec = vorticity_from_uv(x_rec[..., :2])

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(w_gt)
        ax1.set_title("GT vorticity")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(w_rec)
        ax2.set_title("Recon vorticity")
        ax2.axis("off")
    else:
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(x[..., 0])
        ax1.set_title("GT")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(x_rec[..., 0])
        ax2.set_title("Recon")
        ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(tok)
    ax3.set_title("Token map")
    ax3.axis("off")

    fig.tight_layout()
    out_png = out / "vq_recon.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    # also write a small metadata txt
    (out / "info.txt").write_text(
        f"npz={npz_path}\nframe={frame}\ncodebook={codebook_size}\nembed={embed_dim}\nhidden={hidden}\n"
        f"commit={float(commit):.6f}\ncodebook_loss={float(cb):.6f}\n"
    )

    print("Saved:", out_png)


# ----------------------------
# Demo: rollout
# ----------------------------

def sample_argmax(logits: jnp.ndarray) -> jnp.ndarray:
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def rollout_tokens(dyn_model: TransformerDynamics, dyn_params: dict,
                   tok_in: jnp.ndarray, L_out: int, vocab: int) -> jnp.ndarray:
    """
    Autoregressively generate next-frame token sequence.

    tok_in: [B, L_in]
    Returns tok_out: [B, L_out]
    """
    B, L_in = tok_in.shape
    # BOS (0) + generated tokens
    tok_out = jnp.zeros((B, 0), dtype=jnp.int32)

    for i in range(L_out):
        # teacher forcing style input: BOS + previous generated
        bos = jnp.zeros((B, 1), dtype=jnp.int32)
        tok_tgt_in = jnp.concatenate([bos, tok_out], axis=1)  # [B, 1+i]

        # pad to full length L_in + L_out so positional embeddings line up with training
        # Unfilled positions = 0 (same as BOS)
        need = L_out - tok_tgt_in.shape[1]
        if need > 0:
            tok_tgt_in = jnp.concatenate([tok_tgt_in, jnp.zeros((B, need), dtype=jnp.int32)], axis=1)

        seq = jnp.concatenate([tok_in, tok_tgt_in], axis=1)  # [B, L_in+L_out]

        logits = dyn_model.apply({"params": dyn_params}, seq, train=False)  # [B, L, vocab]
        # pick token at position corresponding to current i within target segment
        log_i = logits[:, L_in + i, :]  # [B, vocab]
        next_tok = sample_argmax(log_i)[:, None]  # [B,1]
        tok_out = jnp.concatenate([tok_out, next_tok], axis=1)

    return tok_out


def demo_rollout(npz_path: str, vq_ckpt: str, dyn_ckpt: str, out_dir: str,
                 start: int, horizon: int, context: int,
                 codebook_size: int, embed_dim: int, hidden: int,
                 d_model: int, n_heads: int, n_layers: int, dropout: float,
                 stats_path: str | None = None):
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

    # Determine token grid size by encoding one frame
    x0 = jnp.array(fields[start][None, ...], dtype=jnp.float32)
    if mean is not None:
        x0 = (x0 - mean) / (std + 1e-6)
    tok0 = vq_encode_tokens(vq_params, x0)  # [1,h,w]
    h_tok, w_tok = int(tok0.shape[1]), int(tok0.shape[2])
    L_in = context * h_tok * w_tok
    L_out = h_tok * w_tok
    max_len = L_in + L_out

    dyn_cfg = DynConfig(
        vocab_size=codebook_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        max_len=max_len,
    )
    dyn_model, dyn_params = load_dyn_params(dyn_cfg, max_len=max_len, ckpt_path=dyn_ckpt)

    codebook, dec_params = get_codebook_and_decoder_params(vq_params)

    # Prepare initial context tokens from GT frames
    ctx_frames = fields[start : start + context]  # [context,H,W,C]
    ctx_tok_list = []
    for t in range(context):
        xt = jnp.array(ctx_frames[t][None, ...], dtype=jnp.float32)
        if mean is not None:
            xt = (xt - mean) / (std + 1e-6)
        ctx_tok_list.append(vq_encode_tokens(vq_params, xt)[0])  # [h,w]
    ctx_tok = jnp.stack(ctx_tok_list, axis=0)[None, ...]  # [1,context,h,w]

    def flatten_ctx(tok_ctx: jnp.ndarray) -> jnp.ndarray:
        # [B,context,h,w] -> [B, L_in]
        return tok_ctx.reshape((tok_ctx.shape[0], -1))

    def unflatten_grid(tok_flat: jnp.ndarray) -> jnp.ndarray:
        # [B,L_out] -> [B,h,w]
        return tok_flat.reshape((tok_flat.shape[0], h_tok, w_tok))

    # Rollout loop: each step predicts next tokens, then shift context
    preds = []
    gt = fields[start + context : start + context + horizon]  # [horizon,H,W,C]

    tok_ctx = ctx_tok  # [1,context,h,w]
    for k in range(horizon):
        tok_in = flatten_ctx(tok_ctx).astype(jnp.int32)  # [1,L_in]
        tok_next_flat = rollout_tokens(dyn_model, dyn_params, tok_in, L_out=L_out, vocab=codebook_size)  # [1,L_out]
        tok_next = unflatten_grid(tok_next_flat)  # [1,h,w]

        # decode
        x_hat = vq_decode_tokens(vq_cfg, dec_params, codebook, tok_next, out_channels=C)  # [1,H,W,C]
        if mean is not None:
            x_hat = x_hat * (std + 1e-6) + mean
        preds.append(np.array(x_hat[0]))

        # shift context: drop oldest, append predicted
        tok_ctx = jnp.concatenate([tok_ctx[:, 1:], tok_next[:, None, :, :]], axis=1)

    preds = np.stack(preds, axis=0)  # [horizon,H,W,C]

    # Build GIF frames: vorticity GT vs pred, plus error heatmap
    frames = []
    for k in range(horizon):
        gt_k = gt[k]
        pr_k = preds[k]

        if C >= 2:
            w_gt = vorticity_from_uv(gt_k[..., :2])
            w_pr = vorticity_from_uv(pr_k[..., :2])
            err = np.abs(w_pr - w_gt)
        else:
            w_gt = gt_k[..., 0]
            w_pr = pr_k[..., 0]
            err = np.abs(w_pr - w_gt)

        fig = plt.figure(figsize=(9, 3))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(w_gt)
        ax1.set_title(f"GT (t={k})")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(w_pr)
        ax2.set_title("Rollout")
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(err)
        ax3.set_title("Abs error")
        ax3.axis("off")

        fig.tight_layout()

        # render figure to array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        frames.append(img)

    out_gif = out / f"gt_vs_rollout_{horizon}.gif"
    imageio.mimsave(out_gif, frames, duration=0.12)

    # save a small npz for analysis later
    np.savez_compressed(out / "rollout_arrays.npz", gt=gt, pred=preds)

    print("Saved:", out_gif)
    print(f"Token grid: {h_tok}x{w_tok} | context={context} | horizon={horizon} | max_len={max_len}")


# ----------------------------
# Entrypoint
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["tokenizer", "rollout"], required=True)
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--stats", type=str, default="", help="Stats .npz for normalization (mean/std)")

    # tokenizer ckpt
    ap.add_argument("--vq_ckpt", type=str, required=True)

    # tokenizer config (must match training)
    ap.add_argument("--codebook", type=int, default=512)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)

    # tokenizer demo
    ap.add_argument("--frame", type=int, default=0)

    # rollout config
    ap.add_argument("--dyn_ckpt", type=str, default="")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--context", type=int, default=2)

    # dynamics model config (must match training)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)

    args = ap.parse_args()

    if args.mode == "tokenizer":
        demo_tokenizer(
            npz_path=args.npz,
            vq_ckpt=args.vq_ckpt,
            out_dir=args.out,
            frame=args.frame,
            codebook_size=args.codebook,
            embed_dim=args.embed,
            hidden=args.hidden,
            stats_path=args.stats if args.stats else None,
        )
        return

    # rollout mode
    if not args.dyn_ckpt:
        raise ValueError("--dyn_ckpt is required for --mode rollout")

    demo_rollout(
        npz_path=args.npz,
        vq_ckpt=args.vq_ckpt,
        dyn_ckpt=args.dyn_ckpt,
        out_dir=args.out,
        start=args.start,
        horizon=args.horizon,
        context=args.context,
        codebook_size=args.codebook,
        embed_dim=args.embed,
        hidden=args.hidden,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        stats_path=args.stats if args.stats else None,
    )


if __name__ == "__main__":
    main()
