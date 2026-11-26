"""Layers for fields on a periodic two-dimensional domain."""

from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import linen as nn


class PeriodicConv(nn.Module):
    features: int
    strides: tuple[int, int] = (1, 1)
    zero_init: bool = False

    @nn.compact
    def __call__(self, x):
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")
        return nn.Conv(
            self.features,
            (3, 3),
            strides=self.strides,
            padding="VALID",
            kernel_init=nn.initializers.zeros_init()
            if self.zero_init
            else nn.initializers.lecun_normal(),
        )(x)


def groups(channels: int) -> int:
    return next(n for n in range(min(8, channels), 0, -1) if channels % n == 0)


def upsample(x):
    return jax.image.resize(
        x, (x.shape[0], 2 * x.shape[1], 2 * x.shape[2], x.shape[-1]), method="nearest"
    )
