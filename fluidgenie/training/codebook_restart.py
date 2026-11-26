"""Restart unused cosine codes from observed encoder embeddings."""

import jax
import jax.numpy as jnp
from flax.core import unfreeze, freeze, FrozenDict


def restart_codes(state, usage, candidates, rng, reserve_code0=False):
    params = unfreeze(state.params)
    codebook = params["vq"]["codebook"]
    candidates = candidates.astype(codebook.dtype).reshape(-1, codebook.shape[-1])
    if candidates.shape[0] == 0:
        return state, jnp.asarray(0, dtype=jnp.int32)
    selection_key, jitter_key = jax.random.split(rng)
    selected = jax.random.choice(
        selection_key,
        candidates.shape[0],
        (codebook.shape[0],),
        replace=candidates.shape[0] < codebook.shape[0],
    )
    replacements = candidates[selected] + 1e-4 * jax.random.normal(
        jitter_key, codebook.shape, dtype=codebook.dtype
    )
    replacements /= jnp.maximum(
        jnp.linalg.norm(replacements, axis=-1, keepdims=True), 1e-8
    )
    dead = usage == 0
    if reserve_code0:
        dead = dead.at[0].set(False)
    params["vq"]["codebook"] = jnp.where(dead[:, None], replacements, codebook)

    def clear_dead_slots(path, value):
        if (
            any(getattr(k, "key", None) == "codebook" for k in path)
            and getattr(value, "shape", None) == codebook.shape
        ):
            return jnp.where(dead[:, None], 0, value)
        return value

    opt_state = jax.tree_util.tree_map_with_path(clear_dead_slots, state.opt_state)
    return state.replace(
        params=freeze(params) if isinstance(state.params, FrozenDict) else params,
        opt_state=opt_state,
    ), jnp.sum(dead)


def codebook_health(counts):
    total = jnp.maximum(jnp.sum(counts), 1)
    probabilities = counts / total
    entropy = -jnp.sum(
        jnp.where(
            probabilities > 0,
            probabilities * jnp.log(jnp.maximum(probabilities, 1e-12)),
            0,
        )
    )
    return {"codes_used": jnp.sum(counts > 0), "code_perplexity": jnp.exp(entropy)}
