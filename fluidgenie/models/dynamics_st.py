from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array

from fluidgenie.models.st_transformer import STTransformer


class DynamicsSTMaskGIT(nn.Module):
    """Spatial-temporal MaskGIT dynamics model (optional pipeline)."""

    model_dim: int
    num_latents: int
    num_blocks: int
    num_heads: int
    dropout: float
    mask_ratio_min: float
    mask_ratio_max: float

    def setup(self):
        self.dynamics = STTransformer(
            self.model_dim,
            self.num_latents,
            self.num_blocks,
            self.num_heads,
            self.dropout,
        )
        self.patch_embed = nn.Embed(self.num_latents, self.model_dim)
        self.mask_token = self.param(
            "mask_token",
            nn.initializers.lecun_uniform(),
            (1, 1, 1, self.model_dim),
        )
        self.action_up = nn.Dense(self.model_dim)

    def __call__(
        self,
        batch: Dict[str, Any],
        training: bool = True,
    ) -> Dict[str, Array]:
        # video_tokens: [B,T,N] or [B,T,H,W]
        video_tokens = batch["video_tokens"]
        if video_tokens.ndim == 4:
            b, t, h, w = video_tokens.shape
            video_tokens = video_tokens.reshape(b, t, h * w)

        vid_embed = self.patch_embed(video_tokens)

        mask = batch.get("mask", None)
        if mask is not None and mask.ndim == 4:
            b, t, h, w = mask.shape
            mask = mask.reshape(b, t, h * w)
        if mask is None and training:
            rng1, rng2 = jax.random.split(batch["mask_rng"])
            if self.mask_ratio_min >= self.mask_ratio_max:
                mask_prob = self.mask_ratio_max
            else:
                mask_prob = jax.random.uniform(
                    rng1, minval=self.mask_ratio_min, maxval=self.mask_ratio_max
                )
            mask = jax.random.bernoulli(rng2, mask_prob, vid_embed.shape[:-1])
        if mask is not None:
            mask = mask.at[:, 0].set(False)
            vid_embed = jnp.where(mask[..., None], self.mask_token, vid_embed)

        latent_actions = batch.get("latent_actions", None)
        if latent_actions is not None:
            act_embed = self.action_up(latent_actions)
            if act_embed.ndim == 3:
                act_embed = act_embed[:, :, None, :]
            vid_embed += jnp.pad(act_embed, ((0, 0), (1, 0), (0, 0), (0, 0)))

        logits = self.dynamics(vid_embed, training=training)
        return dict(token_logits=logits, mask=mask)
