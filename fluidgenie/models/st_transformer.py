from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float, Int


class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    def setup(self):
        self.pe = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model),
        )

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:
        return x + self.pe[: x.shape[2]]


class STBlock(nn.Module):
    dim: int
    num_heads: int
    dropout: float

    @nn.remat
    @nn.compact
    def __call__(self, x: Float[Array, "b t n d"], training: bool) -> Float[Array, "b t n d"]:
        # Spatial attention over patches
        z = PositionalEncoding(self.dim)(x)
        z = nn.LayerNorm()(z)
        z = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            deterministic=not training,
        )(z)
        x = x + z

        # Temporal attention over frames (causal)
        x = x.swapaxes(1, 2)
        z = PositionalEncoding(self.dim)(x)
        z = nn.LayerNorm()(z)
        causal_mask = jnp.tri(z.shape[-2])
        z = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            deterministic=not training,
        )(z, mask=causal_mask)
        x = x + z
        x = x.swapaxes(1, 2)

        # Feedforward
        z = nn.LayerNorm()(x)
        z = nn.Dense(self.dim)(z)
        z = nn.gelu(z)
        x = x + z
        return x


class STTransformer(nn.Module):
    model_dim: int
    out_dim: int
    num_blocks: int
    num_heads: int
    dropout: float

    @nn.compact
    def __call__(
        self, x: Float[Array, "b t n d_in"], training: bool
    ) -> Float[Array, "b t n d_out"]:
        x = nn.Sequential([nn.LayerNorm(), nn.Dense(self.model_dim), nn.LayerNorm()])(x)
        for _ in range(self.num_blocks):
            x = STBlock(
                dim=self.model_dim, num_heads=self.num_heads, dropout=self.dropout
            )(x, training=training)
        return nn.Dense(self.out_dim)(x)


def normalize(x: Float[Array, "... d"]) -> Float[Array, "... d"]:
    return x / (jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-8)


class VectorQuantizer(nn.Module):
    latent_dim: int
    num_latents: int
    dropout: float

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
        if training:
            distance = self.drop(distance, deterministic=False)

        indices = jnp.argmin(distance, axis=-1)
        z = self.codebook[indices]
        z_q = x + jax.lax.stop_gradient(z - x)
        return z_q, z, x, indices

    def get_codes(self, indices: Int[Array, "..."]) -> Float[Array, "... d"]:
        return self.codebook[indices]
