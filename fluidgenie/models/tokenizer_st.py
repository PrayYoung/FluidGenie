from __future__ import annotations

from typing import Dict, Any, Tuple

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

    def setup(self):
        self.encoder = STTransformer(
            self.model_dim,
            self.latent_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
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
        )

    def __call__(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        h, w = batch["videos"].shape[2:4]
        outputs = self.vq_encode(batch["videos"], training)
        recon = self.decoder(outputs["z_q"])
        outputs["recon"] = unpatchify(recon, self.patch_size, h, w)
        return outputs

    def vq_encode(self, videos: Any, training: bool = True) -> Dict[str, Any]:
        b, t = videos.shape[:2]
        x = patchify(videos, self.patch_size)
        n = x.shape[2]
        x = self.encoder(x)

        x = x.reshape(b * t * n, self.latent_dim)
        z_q, z, emb, indices = self.vq(x, training)
        z_q = z_q.reshape(b, t, n, self.latent_dim)
        indices = indices.reshape(b, t, n)
        return dict(z_q=z_q, z=z, emb=emb, indices=indices)

    def encode_frame(self, x: Any) -> Any:
        # x: [B,H,W,C] -> tokens [B,h,w]
        b, h, w, c = x.shape
        x = x[:, None, ...]
        tokens = self.vq_encode(x, training=False)["indices"]  # [B,1,N]
        h_pad = -h % self.patch_size
        w_pad = -w % self.patch_size
        hn = (h + h_pad) // self.patch_size
        wn = (w + w_pad) // self.patch_size
        return tokens.reshape(b, 1, hn, wn)[:, 0]

    def decode_tokens(self, indices: Any, video_hw: Tuple[int, int]):
        # indices: [B,h,w]
        b, h, w = indices.shape
        z = self.vq.get_codes(indices.reshape(b, 1, h * w))
        recon = self.decoder(z)
        return unpatchify(recon, self.patch_size, *video_hw)
