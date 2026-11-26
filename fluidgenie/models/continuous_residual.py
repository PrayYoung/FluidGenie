"""Deterministic direct state-space residual dynamics baseline (CR0/CR1)."""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float
from fluidgenie.models.periodic import PeriodicConv, groups, upsample


def bounded_residual_delta(
    raw_delta: jnp.ndarray, residual_scales: Sequence[float]
) -> jnp.ndarray:
    """Smooth per-channel residual bounds in normalized state space."""
    if len(residual_scales) != raw_delta.shape[-1]:
        raise ValueError("residual_scales must have one value per state channel")
    if any(scale <= 0 for scale in residual_scales):
        raise ValueError("residual_scales must be strictly positive")
    scales = jnp.asarray(residual_scales, dtype=raw_delta.dtype).reshape(
        1, 1, 1, raw_delta.shape[-1]
    )
    return scales * jnp.tanh(raw_delta / scales)


class ResidualConvBlock(nn.Module):
    features: int
    stable: bool = False

    @nn.compact
    def __call__(self, x: Float[Array, "b h w c"]) -> Float[Array, "b h w c_out"]:
        residual = x
        if self.stable:
            h = PeriodicConv(self.features)(
                nn.gelu(nn.GroupNorm(num_groups=groups(x.shape[-1]))(x))
            )
            h = PeriodicConv(self.features)(
                nn.gelu(nn.GroupNorm(num_groups=groups(self.features))(h))
            )
        else:
            h = nn.gelu(nn.Conv(self.features, (3, 3), padding="SAME")(x))
            h = nn.Conv(self.features, (3, 3), padding="SAME")(h)
        if residual.shape[-1] != self.features:
            residual = nn.Conv(self.features, (1, 1), padding="SAME")(residual)
        return h + residual if self.stable else nn.gelu(h + residual)


class ContinuousResidualUNet(nn.Module):
    """Predict ``delta_x`` from two normalized full-state context frames."""

    channels: int
    context_frames: int = 2
    hidden_dims: Sequence[int] = (64, 128, 256)
    num_res_blocks: int = 2
    residual_scales: Sequence[float] = ()
    stable: bool = False

    @nn.compact
    def __call__(
        self, context: Float[Array, "b t h w c"], training: bool = True
    ) -> Float[Array, "b h w c"]:
        del training
        if context.shape[1] != self.context_frames:
            raise ValueError(
                f"Expected {self.context_frames} context frames, got {context.shape[1]}"
            )
        if context.shape[2] % 4 or context.shape[3] % 4:
            raise ValueError("Residual U-Net requires height and width divisible by 4")
        if len(self.hidden_dims) != 3:
            raise ValueError("Residual U-Net expects exactly three hidden dimensions")
        b, _, h, w, _ = context.shape
        cond = context.transpose(0, 2, 3, 1, 4).reshape(
            b, h, w, self.context_frames * self.channels
        )
        # Include x[t] explicitly alongside the two-frame context. This matches
        # the state-plus-context input width used by FM0 while retaining residual semantics.
        conv = lambda features, stride=1: (
            PeriodicConv(features, (stride, stride))
            if self.stable
            else nn.Conv(features, (3, 3), strides=(stride, stride), padding="SAME")
        )
        h0 = conv(self.hidden_dims[0])(jnp.concatenate([context[:, -1], cond], axis=-1))
        for _ in range(self.num_res_blocks):
            h0 = ResidualConvBlock(self.hidden_dims[0], stable=self.stable)(h0)
        d1 = conv(self.hidden_dims[1], 2)(h0)
        for _ in range(self.num_res_blocks):
            d1 = ResidualConvBlock(self.hidden_dims[1], stable=self.stable)(d1)
        d2 = conv(self.hidden_dims[2], 2)(d1)
        for _ in range(self.num_res_blocks):
            d2 = ResidualConvBlock(self.hidden_dims[2], stable=self.stable)(d2)
        u1 = (
            conv(self.hidden_dims[1])(upsample(d2))
            if self.stable
            else nn.ConvTranspose(
                self.hidden_dims[1], (4, 4), strides=(2, 2), padding="SAME"
            )(d2)
        )
        u1 = jnp.concatenate([u1, d1], axis=-1)
        for _ in range(self.num_res_blocks):
            u1 = ResidualConvBlock(self.hidden_dims[1], stable=self.stable)(u1)
        u0 = (
            conv(self.hidden_dims[0])(upsample(u1))
            if self.stable
            else nn.ConvTranspose(
                self.hidden_dims[0], (4, 4), strides=(2, 2), padding="SAME"
            )(u1)
        )
        u0 = jnp.concatenate([u0, h0], axis=-1)
        for _ in range(self.num_res_blocks):
            u0 = ResidualConvBlock(self.hidden_dims[0], stable=self.stable)(u0)
        raw_delta = (
            PeriodicConv(self.channels, zero_init=True)(u0)
            if self.stable
            else nn.Conv(self.channels, (3, 3), padding="SAME")(u0)
        )
        saturation = jnp.asarray(0.0, dtype=raw_delta.dtype)
        if self.residual_scales:
            saturation = jnp.mean(
                jnp.abs(
                    jnp.tanh(
                        raw_delta
                        / jnp.asarray(self.residual_scales, dtype=raw_delta.dtype)
                    )
                )
                >= 0.9
            )
        self.sow("diagnostics", "tanh_sat_frac", saturation)
        if not self.residual_scales:
            return raw_delta
        # Smoothly cap each normalized-channel residual while retaining the
        # identity map to first order around zero.
        return bounded_residual_delta(raw_delta, self.residual_scales)
