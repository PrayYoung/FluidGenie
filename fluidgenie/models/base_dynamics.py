from dataclasses import dataclass

import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float, Int

@dataclass
class DynConfig:
    vocab_size: int = 1024
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    max_len: int = 4096  # must cover (context*h*w)

class TransformerDynamics(nn.Module):
    cfg: DynConfig

    @nn.compact
    def __call__(
        self,
        tok_seq: Int[Array, "b l"],
        train: bool,
        decode: bool = False,
        causal: bool = True,
    ) -> Float[Array, "b l vocab"]:
        """
        tok_seq: int32 [B, L]
        returns logits: [B, L, vocab]
        """
        x = nn.Embed(self.cfg.vocab_size, self.cfg.d_model)(tok_seq)

        # learned positional embeddings
        pos = self.param("pos_emb", nn.initializers.normal(stddev=0.02), (self.cfg.max_len, self.cfg.d_model))
        if decode:
            cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0))
            pos_slice = pos[cache_index.value]
            x = x + pos_slice[None, None, :]
            cache_index.value = cache_index.value + x.shape[1]
            mask = None
        else:
            x = x + pos[None, : x.shape[1], :]
            mask = nn.make_causal_mask(tok_seq, dtype=jnp.bool_) if causal else None

        x = nn.Dropout(self.cfg.dropout)(x, deterministic=not train)

        for _ in range(self.cfg.n_layers):
            h = nn.LayerNorm()(x)
            h = nn.SelfAttention(
                num_heads=self.cfg.n_heads,
                qkv_features=self.cfg.d_model,
                dropout_rate=self.cfg.dropout,
                deterministic=not train,
                decode=decode,
            )(h, mask=mask)
            x = x + h

            h = nn.LayerNorm()(x)
            h = nn.Dense(self.cfg.d_model * 4)(h)
            h = nn.gelu(h)
            h = nn.Dropout(self.cfg.dropout)(h, deterministic=not train)
            h = nn.Dense(self.cfg.d_model)(h)
            x = x + h

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.cfg.vocab_size)(x)
        return logits
