from __future__ import annotations

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from jaxtyping import Array, Float, Int


def sample_argmax(logits: Float[Array, "b v"]) -> Int[Array, "b"]:
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)


def rollout_tokens_autoregressive(
    dyn_model,
    dyn_params,
    tok_in: Int[Array, "b l_in"],
    L_out: int,
    vocab: int,
    bos_token_id: int = 0,
) -> Int[Array, "b l_out"]:
    B, L_in = tok_in.shape
    tok_out = jnp.zeros((B, 0), dtype=jnp.int32)

    for i in range(L_out):
        bos = jnp.full((B, 1), bos_token_id, dtype=jnp.int32)
        tok_tgt_in = jnp.concatenate([bos, tok_out], axis=1)  # [B, 1+i]

        need = L_out - tok_tgt_in.shape[1]
        if need > 0:
            tok_tgt_in = jnp.concatenate([tok_tgt_in, jnp.zeros((B, need), dtype=jnp.int32)], axis=1)

        seq = jnp.concatenate([tok_in, tok_tgt_in], axis=1)  # [B, L_in+L_out]

        logits = dyn_model.apply({"params": dyn_params}, seq, train=False, causal=True)  # [B, L, vocab]
        log_i = logits[:, L_in + i, :]  # [B, vocab]
        next_tok = sample_argmax(log_i)[:, None]  # [B,1]
        tok_out = jnp.concatenate([tok_out, next_tok], axis=1)

    return tok_out


def rollout_tokens_autoregressive_cached(
    dyn_model,
    dyn_params,
    tok_in: Int[Array, "b l_in"],
    L_out: int,
    bos_token_id: int = 0,
    rng_seed: int = 0,
) -> Int[Array, "b l_out"]:
    """
    Autoregressive rollout using KV cache (decode=True).
    """
    B, L_in = tok_in.shape
    rng = jax.random.PRNGKey(rng_seed)
    variables = dyn_model.init(rng, jnp.zeros((B, 1), dtype=jnp.int32), train=False, decode=True)
    cache = variables["cache"]
    # reset cache index to 0 because init() advances it by 1
    cache_mut = unfreeze(cache)
    cache_mut["cache_index"] = jnp.array(0)
    cache = freeze(cache_mut)

    # prefill cache with context tokens (scan for compilation)
    def prefill_step(carry_cache, token_slice):
        token = token_slice[:, None]
        _, updated = dyn_model.apply(
            {"params": dyn_params, "cache": carry_cache},
            token,
            train=False,
            decode=True,
            mutable=["cache"],
        )
        return freeze(updated["cache"]), None

    cache, _ = jax.lax.scan(prefill_step, cache, tok_in.T)

    # BOS token
    bos = jnp.full((B, 1), bos_token_id, dtype=jnp.int32)
    logits, updated = dyn_model.apply(
        {"params": dyn_params, "cache": cache},
        bos,
        train=False,
        decode=True,
        mutable=["cache"],
    )
    cache = freeze(updated["cache"])

    next_tok = sample_argmax(logits[:, -1, :])

    def gen_step(carry, _):
        carry_cache, prev_tok = carry
        token = prev_tok[:, None]
        logits, updated = dyn_model.apply(
            {"params": dyn_params, "cache": carry_cache},
            token,
            train=False,
            decode=True,
            mutable=["cache"],
        )
        next_t = sample_argmax(logits[:, -1, :])
        return (freeze(updated["cache"]), next_t), next_t

    if L_out <= 1:
        return next_tok[:, None]

    (_, _), tok_rest = jax.lax.scan(gen_step, (cache, next_tok), None, length=L_out - 1)
    tok_out = jnp.concatenate([next_tok[:, None], tok_rest.T], axis=1)
    return tok_out
