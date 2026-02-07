from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes

from fluidgenie.models.vq_tokenizer import VQVAE, VQConfig
from fluidgenie.training.config_utils import load_toml_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Directory with .npz episodes")
    ap.add_argument("--vq_ckpt", type=str, required=True)
    ap.add_argument("--codebook", type=int, default=1024)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--frames", type=int, default=8, help="Frames per episode to sample")
    ap.add_argument("--episodes", type=int, default=20, help="Number of episodes to sample")
    ap.add_argument("--stats", type=str, default="", help="Stats .npz for normalization (mean/std)")
    ap.add_argument("--tokenizer_config", type=str, default="", help="Tokenizer TOML config (for codebook/embed/hidden/stats)")
    args = ap.parse_args()

    if args.tokenizer_config:
        cfg = load_toml_config(args.tokenizer_config, section="tokenizer")
        if "codebook" in cfg:
            args.codebook = cfg["codebook"]
        if "embed" in cfg:
            args.embed = cfg["embed"]
        if "hidden" in cfg:
            args.hidden = cfg["hidden"]
        if not args.stats and "stats" in cfg:
            args.stats = cfg["stats"]

    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))[: args.episodes]
    if not files:
        raise FileNotFoundError(f"No npz files found in {args.data}")

    sample = np.load(files[0])["fields"][0]
    H, W, C = sample.shape

    vq_cfg = VQConfig(codebook_size=args.codebook, embed_dim=args.embed, hidden=args.hidden)
    vq_model = VQVAE(vq_cfg, in_channels=C)
    rng = jax.random.PRNGKey(0)
    init_params = vq_model.init(rng, jnp.zeros((1, H, W, C), dtype=jnp.float32))["params"]
    vq_params = from_bytes(init_params, Path(args.vq_ckpt).read_bytes())

    if args.stats:
        stats = np.load(args.stats)
        mean = stats["mean"].reshape(1, 1, 1, -1)
        std = stats["std"].reshape(1, 1, 1, -1)
    else:
        mean = None
        std = None

    def encode(x):
        if mean is not None:
            x = (x - mean) / (std + 1e-6)
        x = jnp.array(x, dtype=jnp.float32)
        _x_rec, tok, _c, _cb = vq_model.apply({"params": vq_params}, x)
        return np.array(tok)

    all_tok = []
    for f in files:
        fields = np.load(f)["fields"]
        frames = fields[: args.frames]
        tok = encode(frames)
        all_tok.append(tok.reshape(-1))

    all_tok = np.concatenate(all_tok, axis=0)
    hist = np.bincount(all_tok, minlength=args.codebook)
    active = int((hist > 0).sum())
    ratio = active / args.codebook

    print(f"Active tokens: {active}/{args.codebook} ({ratio:.2%})")
    print("Top-10 token counts:", np.sort(hist)[-10:])
    print("Bottom-10 token counts:", np.sort(hist)[:10])


if __name__ == "__main__":
    main()
