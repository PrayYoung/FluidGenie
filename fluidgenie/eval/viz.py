from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from fluidgenie.models.vq_tokenizer import VQConfig
from fluidgenie.eval.utils import ensure_dir, vorticity_from_uv, load_vq_params


def save_tokenizer_recon(
    npz_path: str,
    vq_ckpt: str,
    out_dir: str,
    frame: int,
    codebook_size: int,
    embed_dim: int,
    hidden: int,
    stats_path: Optional[str] = None,
) -> None:
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

    (out / "info.txt").write_text(
        f"npz={npz_path}\nframe={frame}\ncodebook={codebook_size}\nembed={embed_dim}\nhidden={hidden}\n"
        f"commit={float(commit):.6f}\ncodebook_loss={float(cb):.6f}\n"
    )

    print("Saved:", out_png)
