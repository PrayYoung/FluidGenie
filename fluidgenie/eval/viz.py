from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float, Int

from fluidgenie.models.base_tokenizer import VQConfig
from fluidgenie.eval.utils import (
    ensure_dir,
    vorticity_from_uv,
    load_tokenizer_params,
    make_vq_encode_tokens,
    make_st_encode_tokens,
    st_decode_tokens,
)


def save_tokenizer_recon(
    npz_path: str,
    vq_ckpt: str,
    out_dir: str,
    frame: int,
    codebook_size: int,
    embed_dim: int,
    hidden: int,
    stats_path: Optional[str] = None,
    save_gif: bool = False,
    view: str = "density",
    tokenizer_arch: str = "conv",
    patch_size: int = 4,
    model_dim: int = 256,
    num_blocks: int = 6,
    num_heads: int = 8,
    dropout: float = 0.0,
    codebook_dropout: float = 0.0,
) -> None:
    out = ensure_dir(Path(out_dir))

    fields: Float[Array, "t h w c"] = np.load(npz_path, mmap_mode="r")  # [T,H,W,C]
    x = fields[frame]        # [H,W,C]
    H, W, C = x.shape

    vq_cfg = VQConfig(codebook_size=codebook_size, embed_dim=embed_dim, hidden=hidden)
    base_or_st_tokenizer_model, base_or_st_tokenizer_params = load_tokenizer_params(
        tokenizer_arch,
        vq_cfg,
        in_channels=C,
        H=H,
        W=W,
        ckpt_path=vq_ckpt,
        patch_size=patch_size,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout=dropout,
        codebook_dropout=codebook_dropout,
    )
    if tokenizer_arch == "st":
        st_tokenizer_model = base_or_st_tokenizer_model
        st_tokenizer_params = base_or_st_tokenizer_params
        vq_encode_tokens = make_st_encode_tokens(st_tokenizer_model)
    else:
        base_tokenizer_model = base_or_st_tokenizer_model
        base_tokenizer_params = base_or_st_tokenizer_params
        vq_encode_tokens = make_vq_encode_tokens(base_tokenizer_model)

    if stats_path:
        stats = np.load(stats_path)
        mean = stats["mean"].reshape(1, 1, -1)
        std = stats["std"].reshape(1, 1, -1)
        x_norm = (x - mean) / (std + 1e-6)
    else:
        x_norm = x

    x_in = jnp.array(x_norm[None, ...], dtype=jnp.float32)
    if tokenizer_arch == "st":
        tok = vq_encode_tokens(st_tokenizer_params, x_in)[0]
        x_rec = st_decode_tokens(st_tokenizer_model, st_tokenizer_params, tok[None, ...], (H, W))[0]
        commit = 0.0
        cb = 0.0
    else:
        x_rec, tok, commit, cb = base_tokenizer_model.apply(
            {"params": base_tokenizer_params}, x_in
        )
        x_rec = np.array(x_rec[0])
        tok = np.array(tok[0])
    if stats_path:
        x_rec = x_rec * (std + 1e-6) + mean
    tok = np.array(tok)

    fig = plt.figure(figsize=(12, 4))

    if view == "density" and C >= 3:
        gt_vis = x[..., 2]
        rec_vis = x_rec[..., 2]
        title = "Density"
    elif view == "speed" and C >= 2:
        gt_vis = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        rec_vis = np.sqrt(x_rec[..., 0] ** 2 + x_rec[..., 1] ** 2)
        title = "Speed"
    elif view == "vorticity" and C >= 2:
        gt_vis = vorticity_from_uv(x[..., :2])
        rec_vis = vorticity_from_uv(x_rec[..., :2])
        title = "Vorticity"
    else:
        gt_vis = x[..., 0]
        rec_vis = x_rec[..., 0]
        title = "Channel0"

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(gt_vis)
    ax1.set_title(f"GT {title}")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(rec_vis)
    ax2.set_title(f"Recon {title}")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(tok)
    ax3.set_title("Token map")
    ax3.axis("off")

    fig.tight_layout()
    out_png = out / "vq_recon.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    (out / "info.txt").write_text(
        f"npz={npz_path}\nframe={frame}\narch={tokenizer_arch}\ncodebook={codebook_size}\nembed={embed_dim}\n"
        f"hidden={hidden}\npatch_size={patch_size}\nmodel_dim={model_dim}\nnum_blocks={num_blocks}\nnum_heads={num_heads}\n"
        f"commit={float(commit):.6f}\ncodebook_loss={float(cb):.6f}\n"
    )

    print("Saved:", out_png)

    if not save_gif:
        return

    try:
        import imageio.v2 as imageio
    except Exception:  # pragma: no cover
        import imageio  # type: ignore

    frames = []
    for t in range(fields.shape[0]):
        x_t = fields[t]
        x_in = x_t
        if stats_path:
            x_in = (x_t - mean) / (std + 1e-6)
        x_in = jnp.array(x_in[None, ...], dtype=jnp.float32)
        if tokenizer_arch == "st":
            tok = vq_encode_tokens(st_tokenizer_params, x_in)[0]
            x_rec = st_decode_tokens(st_tokenizer_model, st_tokenizer_params, tok[None, ...], (H, W))[0]
        else:
            x_rec, tok, _, _ = base_tokenizer_model.apply(
                {"params": base_tokenizer_params}, x_in
            )
            x_rec = np.array(x_rec[0])
            tok = np.array(tok[0])
        if stats_path:
            x_rec = x_rec * (std + 1e-6) + mean
        tok = np.array(tok)

        fig = plt.figure(figsize=(12, 4))
        if view == "density" and C >= 3:
            gt_vis = x_t[..., 2]
            rec_vis = x_rec[..., 2]
            title = "Density"
        elif view == "speed" and C >= 2:
            gt_vis = np.sqrt(x_t[..., 0] ** 2 + x_t[..., 1] ** 2)
            rec_vis = np.sqrt(x_rec[..., 0] ** 2 + x_rec[..., 1] ** 2)
            title = "Speed"
        elif view == "vorticity" and C >= 2:
            gt_vis = vorticity_from_uv(x_t[..., :2])
            rec_vis = vorticity_from_uv(x_rec[..., :2])
            title = "Vorticity"
        else:
            gt_vis = x_t[..., 0]
            rec_vis = x_rec[..., 0]
            title = "Channel0"

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(gt_vis)
        ax1.set_title(f"GT {title} (t={t})")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(rec_vis)
        ax2.set_title(f"Recon {title}")
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(tok)
        ax3.set_title("Token map")
        ax3.axis("off")

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = np.array(buf)[..., :3]
        plt.close(fig)
        frames.append(img)

    out_gif = out / "vq_recon_all.gif"
    imageio.mimsave(out_gif, frames, duration=0.12)
    print("Saved:", out_gif)
