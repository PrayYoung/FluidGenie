"""ODE samplers for the isolated direct flow-matching baseline."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _initial_noise(context: Float[Array, "b t h w c"], rng: jax.Array) -> jax.Array:
    return jax.random.normal(
        rng, context.shape[:1] + context.shape[2:], dtype=context.dtype
    )


def sample_flow_matching_euler(
    model,
    params,
    context: Float[Array, "b t h w c"],
    rng: jax.Array,
    ode_steps: int,
    action_cond=None,
) -> Float[Array, "b h w c"]:
    """Integrate dx/dtau = v_theta(x, tau, context) with Euler steps."""
    if ode_steps < 1:
        raise ValueError("ode_steps must be positive")
    x0 = _initial_noise(context, rng)
    dt = 1.0 / float(ode_steps)

    def step(x, step_idx):
        tau = jnp.full((x.shape[0],), step_idx * dt, dtype=x.dtype)
        velocity = model.apply(
            {"params": params},
            x,
            tau,
            context,
            training=False,
            **({"action_cond": action_cond} if action_cond is not None else {}),
        )
        return x + dt * velocity, None

    x, _ = jax.lax.scan(step, x0, jnp.arange(ode_steps))
    return x


def sample_flow_matching_heun(
    model,
    params,
    context: Float[Array, "b t h w c"],
    rng: jax.Array,
    ode_steps: int,
    action_cond=None,
) -> Float[Array, "b h w c"]:
    """Heun on interior steps, then Euler without evaluating the tau=1 endpoint."""
    if ode_steps < 1:
        raise ValueError("ode_steps must be positive")
    x0 = _initial_noise(context, rng)
    dt = 1.0 / float(ode_steps)

    def step(x, step_idx):
        tau = jnp.full((x.shape[0],), step_idx * dt, dtype=x.dtype)
        v1 = model.apply(
            {"params": params},
            x,
            tau,
            context,
            training=False,
            **({"action_cond": action_cond} if action_cond is not None else {}),
        )
        proposal = x + dt * v1
        tau_next = jnp.full((x.shape[0],), (step_idx + 1) * dt, dtype=x.dtype)
        v2 = model.apply(
            {"params": params},
            proposal,
            tau_next,
            context,
            training=False,
            **({"action_cond": action_cond} if action_cond is not None else {}),
        )
        return x + 0.5 * dt * (v1 + v2), None

    x, _ = jax.lax.scan(step, x0, jnp.arange(ode_steps - 1))
    tau = jnp.full((x.shape[0],), (ode_steps - 1) * dt, dtype=x.dtype)
    v = model.apply(
        {"params": params},
        x,
        tau,
        context,
        training=False,
        **({"action_cond": action_cond} if action_cond is not None else {}),
    )
    return x + dt * v
