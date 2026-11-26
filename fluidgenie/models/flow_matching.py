"""Minimal direct state-space conditional flow-matching model.

This module is intentionally independent of the tokenizer and discrete dynamics
pipelines. It models a full normalized fluid field directly.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float
from fluidgenie.models.periodic import PeriodicConv, groups, upsample


class SinusoidalTimeEmbedding(nn.Module):
    dim: int
    modern: bool = False

    @nn.compact
    def __call__(self, tau: Float[Array, "b"]) -> Float[Array, "b d"]:
        half = max(1, self.dim // 2)
        freqs = (
            jnp.exp(jnp.linspace(0.0, -jnp.log(10_000.0), half, dtype=tau.dtype))
            if self.modern
            else jnp.exp(jnp.linspace(0.0, jnp.log(10_000.0), half, dtype=tau.dtype))
        )
        angles = tau[:, None] * (1000.0 if self.modern else 1.0) * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        emb = nn.Dense(self.dim)(emb)
        return nn.Dense(self.dim)(nn.silu(emb)) if self.modern else emb


class ResidualBlock(nn.Module):
    features: int
    modern: bool = False

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "b h w c"],
        time_emb: Float[Array, "b d"],
    ) -> Float[Array, "b h w c_out"]:
        residual = x
        if self.modern:
            h = PeriodicConv(self.features)(
                nn.silu(nn.GroupNorm(num_groups=groups(x.shape[-1]))(x))
            )
            scale, shift = jnp.split(
                nn.Dense(
                    2 * self.features,
                    kernel_init=nn.initializers.zeros_init(),
                    bias_init=nn.initializers.zeros_init(),
                )(nn.silu(time_emb)),
                2,
                axis=-1,
            )
            h = nn.GroupNorm(num_groups=groups(self.features))(h)
            h = h * (1 + scale[:, None, None]) + shift[:, None, None]
            h = PeriodicConv(self.features)(nn.silu(h))
        else:
            h = nn.Conv(self.features, (3, 3), padding="SAME")(x)
            h = h + nn.Dense(self.features)(time_emb)[:, None, None, :]
            h = nn.gelu(h)
            h = nn.Conv(self.features, (3, 3), padding="SAME")(h)
        if residual.shape[-1] != self.features:
            residual = nn.Conv(self.features, (1, 1), padding="SAME")(residual)
        return h + residual if self.modern else nn.gelu(h + residual)


class ConditionalFlowUNet(nn.Module):
    """Small U-Net vector field conditioned on the two most recent fields."""

    channels: int
    context_frames: int = 2
    hidden_dims: Sequence[int] = (64, 128, 256)
    time_dim: int = 128
    num_res_blocks: int = 1
    modern: bool = False
    action_dim: int = 0

    @nn.compact
    def __call__(
        self,
        x_tau: Float[Array, "b h w c"],
        tau: Float[Array, "b"],
        context: Float[Array, "b t h w c"],
        training: bool = True,
        action_cond=None,
    ) -> Float[Array, "b h w c"]:
        del training  # no dropout/batch-statistics in FM0
        if context.shape[1] != self.context_frames:
            raise ValueError(
                f"Expected {self.context_frames} context frames, got {context.shape[1]}"
            )
        if x_tau.shape[1] % 4 or x_tau.shape[2] % 4:
            raise ValueError("FM0 U-Net requires height and width divisible by 4")
        if len(self.hidden_dims) != 3:
            raise ValueError("FM0 expects exactly three hidden dimensions")

        b, h, w, _ = x_tau.shape
        cond = context.transpose(0, 2, 3, 1, 4).reshape(
            b, h, w, self.context_frames * self.channels
        )
        time_emb = SinusoidalTimeEmbedding(self.time_dim, modern=self.modern)(tau)
        if self.action_dim:
            if action_cond is None or action_cond.shape != (b, self.action_dim):
                raise ValueError("action_cond must have shape [batch, action_dim]")
            time_emb = time_emb + nn.Dense(self.time_dim)(
                jnp.asarray(action_cond, dtype=x_tau.dtype)
            )
        conv = lambda features, stride=1: (
            PeriodicConv(features, (stride, stride))
            if self.modern
            else nn.Conv(features, (3, 3), strides=(stride, stride), padding="SAME")
        )
        h0 = conv(self.hidden_dims[0])(jnp.concatenate([x_tau, cond], axis=-1))
        for _ in range(self.num_res_blocks):
            h0 = ResidualBlock(self.hidden_dims[0], modern=self.modern)(h0, time_emb)

        d1 = conv(self.hidden_dims[1], 2)(h0)
        for _ in range(self.num_res_blocks):
            d1 = ResidualBlock(self.hidden_dims[1], modern=self.modern)(d1, time_emb)

        d2 = conv(self.hidden_dims[2], 2)(d1)
        for _ in range(self.num_res_blocks):
            d2 = ResidualBlock(self.hidden_dims[2], modern=self.modern)(d2, time_emb)

        if self.modern:
            b2, h2, w2, c2 = d2.shape
            z = d2.reshape(b2, h2 * w2, c2)
            d2 = (
                z
                + nn.MultiHeadDotProductAttention(num_heads=groups(c2))(
                    nn.LayerNorm()(z)
                )
            ).reshape(d2.shape)
        u1 = (
            conv(self.hidden_dims[1])(upsample(d2))
            if self.modern
            else nn.ConvTranspose(
                self.hidden_dims[1], (4, 4), strides=(2, 2), padding="SAME"
            )(d2)
        )
        u1 = jnp.concatenate([u1, d1], axis=-1)
        for _ in range(self.num_res_blocks):
            u1 = ResidualBlock(self.hidden_dims[1], modern=self.modern)(u1, time_emb)

        u0 = (
            conv(self.hidden_dims[0])(upsample(u1))
            if self.modern
            else nn.ConvTranspose(
                self.hidden_dims[0], (4, 4), strides=(2, 2), padding="SAME"
            )(u1)
        )
        u0 = jnp.concatenate([u0, h0], axis=-1)
        for _ in range(self.num_res_blocks):
            u0 = ResidualBlock(self.hidden_dims[0], modern=self.modern)(u0, time_emb)
        return (
            PeriodicConv(self.channels, zero_init=True)(
                nn.silu(nn.GroupNorm(num_groups=groups(u0.shape[-1]))(u0))
            )
            if self.modern
            else nn.Conv(self.channels, (3, 3), padding="SAME")(u0)
        )
