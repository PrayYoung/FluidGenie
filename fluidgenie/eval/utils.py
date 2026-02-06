from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes

from fluidgenie.models.vq_tokenizer import VQVAE, VQConfig, Decoder
from fluidgenie.models.transformer_dynamics import TransformerDynamics, DynConfig


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def vorticity_from_uv(uv: np.ndarray) -> np.ndarray:
    """uv: [H,W,2] -> vorticity [H,W] using finite differences."""
    u = uv[..., 0]
    v = uv[..., 1]
    dudY = np.gradient(u, axis=0)
    dvdX = np.gradient(v, axis=1)
    return dvdX - dudY


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


def make_vq_encode_tokens(vq_model: VQVAE):
    @jax.jit
    def _encode(vq_params: dict, x: jnp.ndarray) -> jnp.ndarray:
        _x_rec, tok, _commit, _cb = vq_model.apply({"params": vq_params}, x)
        return tok.astype(jnp.int32)
    return _encode


@jax.jit
def vq_decode_tokens(vq_cfg: VQConfig, dec_params: dict, codebook: jnp.ndarray, tok: jnp.ndarray, out_channels: int) -> jnp.ndarray:
    z_q = codebook[tok]  # [B,h,w,D]
    decoder = Decoder(vq_cfg, out_channels=out_channels)
    x_hat = decoder.apply({"params": dec_params}, z_q)
    return x_hat
