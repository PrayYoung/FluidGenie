from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import tyro
from jaxtyping import Array, Float, Int

from fluidgenie.models.base_tokenizer import VQVAE, VQConfig
from fluidgenie.models.tokenizer_st import TokenizerSTVQVAE
from fluidgenie.training.checkpoint_utils import load_params
from configs.eval_configs import EvalCodebookArgs


def main() -> None:
    args = tyro.cli(EvalCodebookArgs)

    files = sorted(glob.glob(os.path.join(args.data, "*.npy")))[: args.episodes]
    if not files:
        raise FileNotFoundError(f"No npy files found in {args.data}")

    sample: Float[Array, "h w c"] = np.load(files[0], mmap_mode="r")[0]
    H, W, C = sample.shape

    rng = jax.random.PRNGKey(args.seed)
    if args.tokenizer_arch == "st":
        st_tokenizer_model = TokenizerSTVQVAE(
            in_dim=C,
            model_dim=args.model_dim,
            latent_dim=args.embed,
            num_latents=args.codebook,
            patch_size=args.patch_size,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout=args.dropout,
            codebook_dropout=args.codebook_dropout,
        )
        init_params = st_tokenizer_model.init(
            rng,
            {"videos": jnp.zeros((1, 1, H, W, C), dtype=jnp.float32)},
            training=False,
        )["params"]
    else:
        vq_cfg = VQConfig(codebook_size=args.codebook, embed_dim=args.embed, hidden=args.hidden)
        base_tokenizer_model = VQVAE(vq_cfg, in_channels=C)
        init_params = base_tokenizer_model.init(
            rng, jnp.zeros((1, H, W, C), dtype=jnp.float32)
        )["params"]
    tokenizer_params = load_params(args.vq_ckpt, init_params)

    if args.stats:
        stats = np.load(args.stats)
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

    def encode(x: Float[Array, "t h w c"]) -> Int[Array, "t h2 w2"]:
        if min_v is not None and denom is not None:
            x = (x - min_v) / denom * 2.0 - 1.0
        elif mean is not None:
            x = (x - mean) / (std + 1e-6)
        x = jnp.array(x, dtype=jnp.float32)
        if args.tokenizer_arch == "st":
            tok = jax.vmap(
                lambda f: st_tokenizer_model.apply(
                    {"params": tokenizer_params}, f[None, ...], method=TokenizerSTVQVAE.encode_frame
                )[0]
            )(x)
        else:
            _x_rec, tok, _c, _cb = base_tokenizer_model.apply(
                {"params": tokenizer_params}, x
            )
        return np.array(tok)

    all_tok = []
    for f in files:
        fields = np.load(f, mmap_mode="r")
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
