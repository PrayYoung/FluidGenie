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
) -> Int[Array, "b l_out"]:
    B, L_in = tok_in.shape

    def step_fn(tok_out, step):
        seq = jnp.concatenate([tok_in, tok_out], axis=1)
        logits = dyn_model.apply({"params": dyn_params}, seq, train=False, causal=False)
        logits_tgt = logits[:, L_in:, :]
        probs = jax.nn.softmax(logits_tgt, axis=-1)
        pred = jnp.argmax(probs, axis=-1).astype(jnp.int32)
        conf = jnp.max(probs, axis=-1)

        # mask schedule (cosine): reveal from few -> many
        t = (step + 1) / mask_steps
        ratio = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
        k = jnp.maximum(1, jnp.floor((1.0 - ratio) * L_out).astype(jnp.int32))

        def update_one(conf_row, pred_row, tok_row):
            # dynamic top-k via threshold (avoids top_k requiring static k)
            sorted_conf = jnp.sort(conf_row)
            kth = sorted_conf[-k]
            keep_mask = conf_row >= kth
            tok_row = jnp.where(keep_mask, pred_row, tok_row)
            return tok_row, keep_mask

        tok_out, _ = jax.vmap(update_one)(conf, pred, tok_out)
        return tok_out, None

    tok_init = jnp.full((B, L_out), mask_token_id, dtype=jnp.int32)
    tok_out, _ = jax.lax.scan(step_fn, tok_init, jnp.arange(mask_steps))
    return tok_out


def st_maskgit_rollout_tokens(
    dyn_model,
    dyn_params,
    tok_ctx: Int[Array, "b t h w"],
    mask_steps: int,
    rng_key: jax.Array,
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
    mask0 = jnp.ones((b, h, w), dtype=jnp.bool_)

    def step_fn(carry, step_idx):
        tok_next, mask, rng = carry
        tok_seq = jnp.concatenate([tok_ctx, tok_next[:, None]], axis=1)  # [B, T+1, h, w]
        step_key, rng = jax.random.split(rng)
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
        # decoding progress
        t_ratio = (step_idx + 1) / mask_steps
        ratio = 0.5 * (1.0 + jnp.cos(jnp.pi * t_ratio))
        # 'k': the absolute number of tokens to UNMASK (keep) at this step
        k = jnp.maximum(1, jnp.floor((1.0 - ratio) * n).astype(jnp.int32))
        # ensure tokens we masked in previous step are never masked again
        # history protection
        mask_flat = mask.reshape(b, n)
        conf_protected = jnp.where(~mask_flat, 1e9, conf)
        # compare all confidence scores against the threshold
        kth = jnp.sort(conf_protected, axis=-1)[:, -k]
        is_unmasked = conf >= kth[:, None]
        # invert 'is_unmask' to create the new mask for the next step
        tok_next_flat = jnp.where(is_unmasked, pred, tok_next.reshape(b, n))
        tok_next = tok_next_flat.reshape(b, h, w)
        mask = (~is_unmasked).reshape(b, h, w)
        if bg_mask is not None:
            tok_next = jnp.where(bg_mask, 0, tok_next)
            mask = jnp.where(bg_mask, False, mask)
        return (tok_next, mask, rng), None

    (tok_next, _, _), _ = jax.lax.scan(step_fn, (tok_next0, mask0, rng_key), jnp.arange(mask_steps))
    return tok_next
