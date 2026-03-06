from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


def maskgit_rollout_tokens(
    dyn_model,
    dyn_params,
    tok_in: Int[Array, "b l_in"],
    L_out: int,
    vocab: int,
    mask_token_id: int,
    mask_steps: int,
    rng_key: jax.Array | None = None,
) -> Int[Array, "b l_out"]:
    B, L_in = tok_in.shape

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    def step_fn(carry, step):
        tok_out, rng = carry
        seq = jnp.concatenate([tok_in, tok_out], axis=1)
        logits = dyn_model.apply({"params": dyn_params}, seq, train=False, causal=False)
        logits_tgt = logits[:, L_in:, :]
        probs = jax.nn.softmax(logits_tgt, axis=-1)
        pred = jnp.argmax(probs, axis=-1).astype(jnp.int32)
        conf = jnp.max(probs, axis=-1)
        rng, noise_key = jax.random.split(rng)
        # Tiny noise only for tie-breaking in equal-confidence regions.
        tie_noise = 1e-6 * jax.random.uniform(noise_key, shape=conf.shape, minval=0.0, maxval=1.0)
        conf_tiebreak = conf + tie_noise

        # mask schedule (cosine): reveal from few -> many
        t = (step + 1) / mask_steps
        ratio = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
        k = jnp.maximum(1, jnp.floor((1.0 - ratio) * L_out).astype(jnp.int32))

        def update_one(conf_row, pred_row, tok_row):
            # Strict top-k reveal for stable denoising step sizes.
            order = jnp.argsort(-conf_row)
            ranks = jnp.empty_like(order)
            ranks = ranks.at[order].set(jnp.arange(conf_row.shape[0]))
            keep_mask = ranks < k
            tok_row = jnp.where(keep_mask, pred_row, tok_row)
            return tok_row, keep_mask

        tok_out, _ = jax.vmap(update_one)(conf_tiebreak, pred, tok_out)
        return (tok_out, rng), None

    tok_init = jnp.full((B, L_out), mask_token_id, dtype=jnp.int32)
    (tok_out, _), _ = jax.lax.scan(step_fn, (tok_init, rng_key), jnp.arange(mask_steps))
    return tok_out


def st_maskgit_rollout_tokens(
    dyn_model,
    dyn_params,
    tok_ctx: Int[Array, "b t h w"],
    mask_steps: int,
    rng_key: jax.Array,
    init_mask_ratio: float = 1.0,
    latent_actions: Float[Array, "b t m d"] | None = None,
    bg_mask: Bool[Array, "b h w"] | None = None,
) -> Int[Array, "b h w"]:
    """
    tok_ctx: [B, context, h, w]
    returns tok_next: [B, h, w]
    """
    b, _, h, w = tok_ctx.shape
    n = h * w
    tok_next0 = jnp.zeros((b, h, w), dtype=jnp.int32)
    init_key, loop_key = jax.random.split(rng_key)
    init_ratio = jnp.clip(jnp.asarray(init_mask_ratio, dtype=jnp.float32), 0.0, 1.0)
    mask0 = jax.random.bernoulli(init_key, init_ratio, shape=(b, h, w))
    if bg_mask is not None:
        mask0 = jnp.where(bg_mask, False, mask0)

    def step_fn(carry, step_idx):
        tok_next, mask, rng = carry
        tok_seq = jnp.concatenate([tok_ctx, tok_next[:, None]], axis=1)  # [B, T+1, h, w]
        step_key, rng = jax.random.split(rng)
        tie_key, rng = jax.random.split(rng)
        batch = {
            "video_tokens": tok_seq,
            "mask_rng": step_key,
            "mask": jnp.concatenate([jnp.zeros_like(tok_ctx, dtype=jnp.bool_), mask[:, None]], axis=1),
        }
        if latent_actions is not None:
            batch["latent_actions"] = latent_actions
        logits = dyn_model.apply({"params": dyn_params}, batch, training=False)
        logits_last = logits["token_logits"][:, -1]  # [B, N, vocab]
        probs = jax.nn.softmax(logits_last, axis=-1)
        pred = jnp.argmax(probs, axis=-1).astype(jnp.int32)
        conf = jnp.max(probs, axis=-1)
        tie_noise = 1e-6 * jax.random.uniform(tie_key, shape=conf.shape, minval=0.0, maxval=1.0)
        # decoding progress
        t_ratio = (step_idx + 1) / mask_steps
        ratio = 0.5 * (1.0 + jnp.cos(jnp.pi * t_ratio))
        # 'k': the absolute number of tokens to UNMASK (keep) at this step
        k = jnp.maximum(1, jnp.floor((1.0 - ratio) * n).astype(jnp.int32))
        # ensure tokens we masked in previous step are never masked again
        # history protection
        mask_flat = mask.reshape(b, n)
        conf_protected = jnp.where(~mask_flat, 1e9, conf + tie_noise)
        # Strict top-k reveal among still-masked positions.
        def update_one(conf_row):
            order = jnp.argsort(-conf_row)
            ranks = jnp.empty_like(order)
            ranks = ranks.at[order].set(jnp.arange(conf_row.shape[0]))
            return ranks < k

        is_unmasked = jax.vmap(update_one)(conf_protected)
        # invert 'is_unmask' to create the new mask for the next step
        tok_next_flat = jnp.where(is_unmasked, pred, tok_next.reshape(b, n))
        tok_next = tok_next_flat.reshape(b, h, w)
        mask = (~is_unmasked).reshape(b, h, w)
        if bg_mask is not None:
            tok_next = jnp.where(bg_mask, 0, tok_next)
            mask = jnp.where(bg_mask, False, mask)
        return (tok_next, mask, rng), None

    (tok_next, _, _), _ = jax.lax.scan(step_fn, (tok_next0, mask0, loop_key), jnp.arange(mask_steps))
    return tok_next
