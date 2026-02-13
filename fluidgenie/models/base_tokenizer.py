from dataclasses import dataclass

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn
from jaxtyping import Array, Float, Int

@dataclass
class VQConfig:
    codebook_size: int = 1024
    embed_dim: int = 64
    hidden: int = 128

class Encoder(nn.Module):
    cfg: VQConfig
    @nn.compact
    def __call__(self, x: Float[Array, "b h w c"]) -> Float[Array, "b h2 w2 d"]:
        # x: [B,H,W,C]
        h = nn.Conv(self.cfg.hidden, (4,4), strides=(2,2), padding="SAME")(x)
        h = nn.gelu(h)
        h = nn.Conv(self.cfg.hidden, (4,4), strides=(2,2), padding="SAME")(h)
        h = nn.gelu(h)
        h = nn.Conv(self.cfg.embed_dim, (1,1), padding="SAME")(h)
        return h  # [B,h,w,embed_dim]

class Decoder(nn.Module):
    cfg: VQConfig
    out_channels: int
    @nn.compact
    def __call__(self, z: Float[Array, "b h w d"]) -> Float[Array, "b h2 w2 c"]:
        h = nn.ConvTranspose(self.cfg.hidden, (4,4), strides=(2,2), padding="SAME")(z)
        h = nn.gelu(h)
        h = nn.ConvTranspose(self.cfg.hidden, (4,4), strides=(2,2), padding="SAME")(h)
        h = nn.gelu(h)
        x = nn.Conv(self.out_channels, (1,1), padding="SAME")(h)
        return x

class VectorQuantizer(nn.Module):
    cfg: VQConfig
    @nn.compact
    def __call__(
        self, z_e: Float[Array, "b h w d"]
    ) -> tuple[
        Float[Array, "b h w d"],
        Int[Array, "b h w"],
        Float[Array, ""],
        Float[Array, ""],
    ]:
        # z_e: [B,h,w,D]
        D = self.cfg.embed_dim
        K = self.cfg.codebook_size

        codebook = self.param("codebook", nn.initializers.uniform(scale=1.0), (K, D))

        z_flat = rearrange(z_e, "b h w d -> (b h w) d")
        # distances: ||z - e||^2 = z^2 + e^2 - 2 z·e
        z2 = jnp.sum(z_flat**2, axis=1, keepdims=True)
        e2 = jnp.sum(codebook**2, axis=1)[None, :]
        ze = 2.0 * (z_flat @ codebook.T)
        dist = z2 + e2 - ze

        idx = jnp.argmin(dist, axis=1)  # [(b*h*w)]
        z_q = codebook[idx]             # [(b*h*w), D]
        z_q = rearrange(z_q, "(b h w) d -> b h w d", b=z_e.shape[0], h=z_e.shape[1], w=z_e.shape[2])

        # straight-through estimator
        z_st = z_e + jax.lax.stop_gradient(z_q - z_e)

        # commitment loss (codebook loss included symmetrically)
        commit = jnp.mean((jax.lax.stop_gradient(z_q) - z_e) ** 2)
        codebook_loss = jnp.mean((z_q - jax.lax.stop_gradient(z_e)) ** 2)

        return z_st, idx.reshape(z_e.shape[0], z_e.shape[1], z_e.shape[2]), commit, codebook_loss

class VQVAE(nn.Module):
    cfg: VQConfig
    in_channels: int
    @nn.compact
    def __call__(
        self, x: Float[Array, "b h w c"]
    ) -> tuple[
        Float[Array, "b h w c"],
        Int[Array, "b h2 w2"],
        Float[Array, ""],
        Float[Array, ""],
    ]:
        z_e = Encoder(self.cfg)(x)
        z_q, tok, commit, codebook_loss = VectorQuantizer(self.cfg)(z_e)
        x_rec = Decoder(self.cfg, out_channels=self.in_channels)(z_q)
        return x_rec, tok, commit, codebook_loss
