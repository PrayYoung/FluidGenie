from __future__ import annotations
import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float, Int


class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    @nn.compact
    def __call__(self, x):
        pe = self.param(
            "pos_emb", nn.initializers.normal(0.02), (self.max_len, self.d_model)
        )
        if x.shape[-2] > self.max_len:
            raise ValueError("Temporal sequence exceeds positional encoding capacity")
        return x + pe[: x.shape[-2]]


class Periodic2DPositionalEncoding(nn.Module):
    d_model: int
    grid_shape: tuple[int, int]
    prefix_tokens: int = 0
    version: int = 3

    @nn.compact
    def __call__(self, x):
        h, w = self.grid_shape
        if min(h, w) < 1 or self.prefix_tokens < 0:
            raise ValueError("Grid dimensions must be positive and prefix nonnegative")
        if x.shape[-2] != h * w + self.prefix_tokens:
            raise ValueError("Patch count does not match periodic grid and prefix")
        y, xx = jnp.meshgrid(
            jnp.arange(h, dtype=x.dtype) / h,
            jnp.arange(w, dtype=x.dtype) / w,
            indexing="ij",
        )
        if self.version not in (2, 3):
            raise ValueError("Unsupported periodic encoding version")
        num_freqs = (
            min(8, min(h, w) // 2)
            if self.version == 3
            else max(1, (self.d_model + 3) // 4)
        )
        harmonics = jnp.arange(1, num_freqs + 1, dtype=x.dtype)
        phase_x = 2 * jnp.pi * xx.reshape(-1, 1) * harmonics
        phase_y = 2 * jnp.pi * y.reshape(-1, 1) * harmonics
        pe = jnp.concatenate(
            (jnp.sin(phase_x), jnp.cos(phase_x), jnp.sin(phase_y), jnp.cos(phase_y)),
            axis=-1,
        )
        if self.version == 2:
            pe = jnp.pad(pe[:, : self.d_model], ((self.prefix_tokens, 0), (0, 0)))
        else:
            pe = nn.Dense(
                self.d_model, name="fourier_projection",
                kernel_init=nn.initializers.lecun_normal() if num_freqs else nn.initializers.zeros_init(),
            )(pe)
            if self.prefix_tokens:
                prefix = self.param(
                    "prefix_embed", nn.initializers.normal(0.02),
                    (self.prefix_tokens, self.d_model),
                )
                pe = jnp.concatenate([prefix, pe], axis=0)
        return x + pe


class STBlock(nn.Module):
    dim: int
    num_heads: int
    dropout: float
    use_causal_mask: bool
    grid_shape: tuple[int, int]
    prefix_tokens: int = 0
    spatial_encoding_version: int = 2

    @nn.compact
    def __call__(self, x, training):
        # Version 4 adds position once at transformer input. Keep the legacy
        # parameter/module layout unchanged when loading version 2/3 models.
        z = x
        if self.spatial_encoding_version != 4:
            z = Periodic2DPositionalEncoding(
                self.dim, self.grid_shape, self.prefix_tokens,
                self.spatial_encoding_version,
            )(x)
        z = nn.LayerNorm()(z)
        x = x + nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=not training,
        )(z)
        x = x.swapaxes(1, 2)
        z = nn.LayerNorm()(
            x if self.spatial_encoding_version == 4 else PositionalEncoding(self.dim)(x)
        )
        mask = (
            nn.make_causal_mask(jnp.ones(z.shape[:-1]))
            if self.use_causal_mask
            else None
        )
        x = x + nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=not training,
        )(z, mask=mask)
        x = x.swapaxes(1, 2)
        z = nn.Dense(4 * self.dim)(nn.LayerNorm()(x))
        z = nn.Dropout(self.dropout)(nn.gelu(z), deterministic=not training)
        return x + nn.Dense(self.dim)(z)


class STTransformer(nn.Module):
    model_dim: int
    out_dim: int
    num_blocks: int
    num_heads: int
    dropout: float
    use_causal_mask: bool = False
    prefix_tokens: int = 0
    zero_init_out: bool = False
    spatial_encoding_version: int = 2

    @nn.compact
    def __call__(self, x, training, spatial_shape=None):
        n = x.shape[2] - self.prefix_tokens
        if spatial_shape is None:
            side = math.isqrt(n)
            if side * side != n:
                raise ValueError("Non-square patch grids require spatial_shape")
            spatial_shape = (side, side)
        if self.spatial_encoding_version not in (2, 3, 4):
            raise ValueError("Unsupported transformer architecture version")
        if self.spatial_encoding_version == 4:
            x = nn.Dense(self.model_dim, name="input_projection")(x)
            x = Periodic2DPositionalEncoding(
                self.model_dim, spatial_shape, self.prefix_tokens, version=3,
                name="spatial_position",
            )(x)
            x = nn.LayerNorm(name="input_norm")(x)
        else:
            x = nn.Sequential([nn.LayerNorm(), nn.Dense(self.model_dim), nn.LayerNorm()])(x)
        for _ in range(self.num_blocks):
            x = STBlock(
                self.model_dim,
                self.num_heads,
                self.dropout,
                self.use_causal_mask,
                spatial_shape,
                self.prefix_tokens,
                self.spatial_encoding_version,
            )(x, training)
        init = (
            nn.initializers.zeros_init()
            if self.zero_init_out
            else nn.initializers.lecun_normal()
        )
        if self.spatial_encoding_version == 4:
            x = nn.LayerNorm(name="output_norm")(x)
        return nn.Dense(self.out_dim, kernel_init=init)(x)


def normalize(x: Float[Array, "... d"]) -> Float[Array, "... d"]:
    return x / (jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-8)


class VectorQuantizer(nn.Module):
    latent_dim: int
    num_latents: int
    dropout: float
    reserve_code0: bool = False

    def setup(self):
        self.codebook = normalize(
            self.param(
                "codebook",
                nn.initializers.lecun_uniform(),
                (self.num_latents, self.latent_dim),
            )
        )
        self.drop = nn.Dropout(self.dropout)

    def __call__(
        self, x: Float[Array, "b d"], training: bool
    ) -> tuple[
        Float[Array, "b d"],
        Float[Array, "b d"],
        Float[Array, "b d"],
        Int[Array, "b"],
    ]:
        x = normalize(x)
        codebook = normalize(self.codebook)
        distance = -jnp.matmul(x, codebook.T)
        if training and self.dropout > 0 and self.has_rng("dropout"):
            mask = jax.random.bernoulli(
                self.make_rng("dropout"), p=1.0 - self.dropout, shape=distance.shape
            )
            distance = jnp.where(mask, distance, 1e9)

        if self.reserve_code0:
            if self.num_latents < 2:
                raise ValueError(
                    "A reserved background code requires at least two codes"
                )
            distance = distance.at[:, 0].set(jnp.inf)
        # Code dropout must retain at least one admissible code in every row.
        fallback = 1 if self.reserve_code0 else 0
        no_code = jnp.all(distance >= 1e9, axis=-1)
        distance = distance.at[:, fallback].set(
            jnp.where(no_code, 0.0, distance[:, fallback])
        )
        indices = jnp.argmin(distance, axis=-1)
        z = self.codebook[indices]
        z_q = x + jax.lax.stop_gradient(z - x)
        return z_q, z, x, indices

    def get_codes(self, indices: Int[Array, "..."]) -> Float[Array, "... d"]:
        return self.codebook[indices]
