"""Scheduled, clipped optimizers shared by the corrected continuous models."""

import optax


def make_optimizer(lr, steps, warmup_steps=200, weight_decay=1e-4, max_grad_norm=1.0):
    if steps < 1 or lr <= 0 or max_grad_norm <= 0 or warmup_steps < 0:
        raise ValueError("Invalid optimizer configuration")
    warmup_steps = min(warmup_steps, steps // 5)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=max(2, steps),
        end_value=lr * 0.01,
    )
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(schedule, weight_decay=weight_decay),
    )
