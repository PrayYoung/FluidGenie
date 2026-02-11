from __future__ import annotations

from typing import Dict, Any

import flax.linen as nn
import jax.numpy as jnp

from fluidgenie.data.preprocess import patchify, unpatchify
from fluidgenie.models.st_transformer import STTransformer, VectorQuantizer


class LatentActionModel(nn.Module):
    """Latent Action ST-ViViT VQ-VAE (optional pipeline)"""

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
        self.patch_token_dim = self.in_dim * self.patch_size**2
        self.encoder = STTransformer(
            self.model_dim,
            self.latent_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
        )
        self.action_in = self.param(
            "action_in",
            nn.initializers.lecun_uniform(),
            (1, 1, 1, self.patch_token_dim),
        )
        self.vq = VectorQuantizer(
            self.latent_dim,
            self.num_latents,
            self.codebook_dropout,
        )
        self.patch_up = nn.Dense(self.model_dim)
        self.action_up = nn.Dense(self.model_dim)
        self.decoder = STTransformer(
            self.model_dim,
            self.patch_token_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
        )

    def __call__(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        h, w = batch["videos"].shape[2:4]
        outputs = self.vq_encode(batch["videos"], training)
        video_action_patches = self.action_up(outputs["z_q"]) + self.patch_up(
            outputs["patches"][:, :-1]
        )
        del outputs["patches"]

        video_recon = self.decoder(video_action_patches)
        outputs["recon"] = unpatchify(video_recon, self.patch_size, h, w)
        return outputs

    def vq_encode(self, videos: Any, training: bool = True) -> Dict[str, Any]:
        b, t = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        action_pad = jnp.broadcast_to(self.action_in, (b, t, 1, self.patch_token_dim))
        padded_patches = jnp.concatenate((action_pad, patches), axis=2)

        z = self.encoder(padded_patches)
        z = z[:, 1:, 0]  # (B, T-1, E)

        z = z.reshape(b * (t - 1), self.latent_dim)
        z_q, z, emb, indices = self.vq(z, training)
        z_q = z_q.reshape(b, t - 1, 1, self.latent_dim)
        return dict(patches=patches, z_q=z_q, z=z, emb=emb, indices=indices)
