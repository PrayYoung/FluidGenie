from __future__ import annotations

from typing import Any, Dict, Tuple

from jaxtyping import Array, Float, Int
import jax.numpy as jnp

import flax.linen as nn
from fluidgenie.data.preprocess import patchify, unpatchify
from fluidgenie.models.st_transformer import STTransformer, VectorQuantizer


class TokenizerSTVQVAE(nn.Module):
    """ST-ViViT VQ-VAE tokenizer (Jafar-style)."""

    in_dim: int
    model_dim: int
    latent_dim: int
    num_latents: int
    patch_size: int
    num_blocks: int
    num_heads: int
    dropout: float
    codebook_dropout: float
    bg_thresh: float = 0.0

    def setup(self):
        self.encoder = STTransformer(
            self.model_dim,
            self.latent_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            # Use causal mask to avoid future leakage and align with autoregressive rollout.
            use_causal_mask=True,
        )
        self.vq = VectorQuantizer(
            self.latent_dim,
            self.num_latents,
            self.codebook_dropout,
        )
        self.out_dim = self.in_dim * self.patch_size**2
        self.decoder = STTransformer(
            self.model_dim,
            self.out_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            # Use causal mask to avoid future leakage and align with autoregressive rollout.
            use_causal_mask=True,
        )

    def __call__(
        self, batch: Dict[str, Any], training: bool = True
    ) -> Dict[str, Array]:
        h, w = batch["videos"].shape[2:4]
        outputs = self.vq_encode(batch["videos"], training)
        recon = self.decoder(outputs["z_q"], training=training)
        outputs["recon"] = unpatchify(recon, self.patch_size, h, w)
        return outputs

    def vq_encode(self, videos: Float[Array, "b t h w c"], training: bool = True) -> Dict[str, Array]:
        b, t = videos.shape[:2]
        x = patchify(videos, self.patch_size)
        n = x.shape[2]
        x = self.encoder(x, training=training)

        x = x.reshape(b * t * n, self.latent_dim)
        z_q, z, emb, indices = self.vq(x, training)
        z_q = z_q.reshape(b, t, n, self.latent_dim)
        indices = indices.reshape(b, t, n)
        return dict(z_q=z_q, z=z, emb=emb, indices=indices)

    def encode_frame(
        self, x: Float[Array, "b h w c"], training: bool = False
    ) -> Int[Array, "b h2 w2"]:
        # x: [B,H,W,C] -> tokens [B,h,w]
        b, h, w, c = x.shape
        x = x[:, None, ...]
        tokens = self.vq_encode(x, training=training)["indices"]  # [B,1,N]
        h_pad = -h % self.patch_size
        w_pad = -w % self.patch_size
        hn = (h + h_pad) // self.patch_size
        wn = (w + w_pad) // self.patch_size
        tok_grid = tokens.reshape(b, 1, hn, wn)[:, 0]
        if self.bg_thresh > 0:
            # For our data, background is near the min value (≈ -1 after min-max).
            # We force pure-vacuum pixels to token 0 to stabilize background dynamics.
            bg_px = jnp.all(jnp.abs(x[:, 0] + 1.0) < self.bg_thresh, axis=-1)  # [B,H,W]
            bg_px = bg_px[:, None, :, :, None].astype(jnp.float32)
            bg_patches = patchify(bg_px, self.patch_size)  # [B,1,N,P]
            bg_patch = jnp.all(bg_patches > 0.5, axis=-1).reshape(b, 1, hn, wn)[:, 0]
            tok_grid = jnp.where(bg_patch, 0, tok_grid)
        return tok_grid

    def decode_tokens(
        self, indices: Int[Array, "b h w"], video_hw: Tuple[int, int]
    ) -> Float[Array, "b h w c"]:
        # indices: [B,h,w]
        b, h, w = indices.shape
        z = self.vq.get_codes(indices.reshape(b, 1, h * w))
        recon = self.decoder(z, training=False)
        img = unpatchify(recon, self.patch_size, *video_hw)
        if self.bg_thresh > 0:
            bg_mask_tok = (indices == 0)
            bg_mask_px = jnp.repeat(jnp.repeat(bg_mask_tok, self.patch_size, axis=1), self.patch_size, axis=2)
            img = jnp.where(bg_mask_px[..., None], -1.0, img)
        if img.ndim == 5:
            return img[:, 0]
        return unpatchify(recon, self.patch_size, *video_hw)
