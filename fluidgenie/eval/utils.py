from __future__ import annotations

from pathlib import Path
from typing import Tuple, Callable, Any

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from fluidgenie.training.checkpoint_utils import load_params

from fluidgenie.models.base_tokenizer import VQVAE, VQConfig, Decoder
from fluidgenie.models.base_dynamics import TransformerDynamics, DynConfig
from fluidgenie.models.tokenizer_st import TokenizerSTVQVAE


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def vorticity_from_uv(uv: Float[Array, "h w 2"]) -> Float[Array, "h w"]:
    """uv: [H,W,2] -> vorticity [H,W] using finite differences."""
    u = uv[..., 0]
    v = uv[..., 1]
    dudY = np.gradient(u, axis=0)
    dvdX = np.gradient(v, axis=1)
    return dvdX - dudY


def load_tokenizer_params(
    arch: str,
    vq_cfg: VQConfig,
    in_channels: int,
    H: int,
    W: int,
    ckpt_path: str,
    seed: int = 0,
    patch_size: int = 4,
    model_dim: int = 256,
    num_blocks: int = 6,
    num_heads: int = 8,
    dropout: float = 0.0,
    codebook_dropout: float = 0.0,
) -> Tuple[Any, dict]:
    rng = jax.random.PRNGKey(seed)
    if arch == "st":
        model = TokenizerSTVQVAE(
            in_dim=in_channels,
            model_dim=model_dim,
            latent_dim=vq_cfg.embed_dim,
            num_latents=vq_cfg.codebook_size,
            patch_size=patch_size,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout,
            codebook_dropout=codebook_dropout,
        )
        params_init = model.init(
            rng,
            {"videos": jnp.zeros((1, 1, H, W, in_channels), dtype=jnp.float32)},
            training=False,
        )["params"]
    else:
        model = VQVAE(vq_cfg, in_channels=in_channels)
        params_init = model.init(rng, jnp.zeros((1, H, W, in_channels), dtype=jnp.float32))["params"]
    params = load_params(ckpt_path, params_init)
    return model, params


def load_dyn_params(dyn_cfg: DynConfig, max_len: int, ckpt_path: str, seed: int = 0):
    model = TransformerDynamics(dyn_cfg)
    rng = jax.random.PRNGKey(seed)
    params_init = model.init(rng, jnp.zeros((1, max_len), dtype=jnp.int32), train=False)["params"]
    params = load_params(ckpt_path, params_init)
    return model, params


def get_codebook_and_decoder_params(vq_params: dict) -> Tuple[Float[Array, "k d"], dict]:
    """
    VQVAE params are typically:
      Encoder_0, VectorQuantizer_0(codebook), Decoder_0
    We keep a fallback search in case module naming differs.
    """
    if "VectorQuantizer_0" in vq_params and "Decoder_0" in vq_params:
        codebook = vq_params["VectorQuantizer_0"]["codebook"]
        dec_params = vq_params["Decoder_0"]
        return codebook, dec_params

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


def make_vq_encode_tokens(vq_model: VQVAE) -> Callable[[dict, Float[Array, "b h w c"]], Int[Array, "b h2 w2"]]:
    @jax.jit
    def _encode(vq_params: dict, x: Float[Array, "b h w c"]) -> Int[Array, "b h2 w2"]:
        _x_rec, tok, _commit, _cb = vq_model.apply({"params": vq_params}, x)
        return tok.astype(jnp.int32)
    return _encode


def make_st_encode_tokens(vq_model: TokenizerSTVQVAE) -> Callable[[dict, Float[Array, "b h w c"]], Int[Array, "b h2 w2"]]:
    @jax.jit
    def _encode(vq_params: dict, x: Float[Array, "b h w c"]) -> Int[Array, "b h2 w2"]:
        tok = vq_model.apply({"params": vq_params}, x, method=TokenizerSTVQVAE.encode_frame)
        return tok.astype(jnp.int32)
    return _encode


def vq_decode_tokens(
    vq_cfg: VQConfig,
    dec_params: dict,
    codebook: Float[Array, "k d"],
    tok: Int[Array, "b h w"],
    out_channels: int,
) -> Float[Array, "b h w c"]:
    z_q = codebook[tok]  # [B,h,w,D]
    decoder = Decoder(vq_cfg, out_channels=out_channels)
    x_hat = decoder.apply({"params": dec_params}, z_q)
    return x_hat


def st_decode_tokens(
    vq_model: TokenizerSTVQVAE,
    vq_params: dict,
    tok: Int[Array, "b h w"],
    video_hw: Tuple[int, int],
) -> Float[Array, "b h w c"]:
    out = vq_model.apply({"params": vq_params}, tok, video_hw, method=TokenizerSTVQVAE.decode_tokens)
    if out.ndim == 5:
        return out[:, 0]
    return out
