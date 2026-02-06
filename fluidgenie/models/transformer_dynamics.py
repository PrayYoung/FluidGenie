from dataclasses import dataclass
import jax.numpy as jnp
from flax import linen as nn

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
    def __call__(self, tok_seq, train: bool):
        """
        tok_seq: int32 [B, L]  (L=context*h*w)
        returns logits: [B, L, vocab]
        """
        x = nn.Embed(self.cfg.vocab_size, self.cfg.d_model)(tok_seq)
        # learned positional embeddings
        pos = self.param("pos_emb", nn.initializers.normal(stddev=0.02), (self.cfg.max_len, self.cfg.d_model))
        x = x + pos[None, : x.shape[1], :]

        x = nn.Dropout(self.cfg.dropout)(x, deterministic=not train)
        mask = nn.make_causal_mask(tok_seq, dtype=jnp.bool_)

        for _ in range(self.cfg.n_layers):
            # self-attention block
            h = nn.LayerNorm()(x)
            h = nn.SelfAttention(num_heads=self.cfg.n_heads, qkv_features=self.cfg.d_model,
                                 dropout_rate=self.cfg.dropout, deterministic=not train)(h, mask=mask)
            x = x + h

            # MLP block
            h = nn.LayerNorm()(x)
            h = nn.Dense(self.cfg.d_model * 4)(h)
            h = nn.gelu(h)
            h = nn.Dropout(self.cfg.dropout)(h, deterministic=not train)
            h = nn.Dense(self.cfg.d_model)(h)
            x = x + h

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.cfg.vocab_size)(x)
        return logits
