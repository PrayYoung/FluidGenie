"""Frozen P1--P3 physical-space auxiliary losses for CR1 screening."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class PhysicsLossConfig:
    """Static, predeclared controls for one-variable CR1 physics ablations."""

    energy_weight: float = 0.0
    sobolev_weight: float = 0.0
    divergence_weight: float = 0.0
    dx: float = 1.0 / 64.0
    energy_floor: float = 0.0
    gradient_ref: float = 1.0
    periodic: bool = True

    @property
    def enabled(self) -> bool:
        return any((self.energy_weight, self.sobolev_weight, self.divergence_weight))


def normalized_to_physical(
    x: jnp.ndarray, minimum: jnp.ndarray, span: jnp.ndarray
) -> jnp.ndarray:
    """Invert the verified dataset's channelwise min/max normalization."""
    return (x + 1.0) * 0.5 * span.reshape(1, 1, 1, -1) + minimum.reshape(1, 1, 1, -1)


def _zero_ghost_derivative(x: jnp.ndarray, axis: int, dx: float) -> jnp.ndarray:
    """Centered derivative with zero ghost cells, matching PhiFlow extrapolation=0."""
    padding = [(0, 0)] * x.ndim
    padding[axis] = (1, 1)
    padded = jnp.pad(x, padding, mode="constant", constant_values=0.0)
    forward = [slice(None)] * x.ndim
    backward = [slice(None)] * x.ndim
    forward[axis] = slice(2, None)
    backward[axis] = slice(None, -2)
    return (padded[tuple(forward)] - padded[tuple(backward)]) / (2.0 * dx)


def periodic_derivative(x, axis, dx):
    if dx <= 0:
        raise ValueError("dx must be positive")
    return (jnp.roll(x, -1, axis=axis) - jnp.roll(x, 1, axis=axis)) / (2 * dx)


def endpoint_physics_losses(
    prediction_normalized: jnp.ndarray,
    target_normalized: jnp.ndarray,
    minimum: jnp.ndarray,
    span: jnp.ndarray,
    cfg: PhysicsLossConfig,
) -> dict[str, jnp.ndarray]:
    """Return unweighted P1/P2/P3 endpoint losses in physical velocity units."""
    prediction = normalized_to_physical(prediction_normalized, minimum, span)
    target = normalized_to_physical(target_normalized, minimum, span)
    u_hat, v_hat = prediction[..., 0], prediction[..., 1]
    u_true, v_true = target[..., 0], target[..., 1]

    derivative = periodic_derivative if cfg.periodic else _zero_ghost_derivative
    du_hat_dx = derivative(u_hat, 2, cfg.dx)
    du_hat_dy = derivative(u_hat, 1, cfg.dx)
    dv_hat_dx = derivative(v_hat, 2, cfg.dx)
    dv_hat_dy = derivative(v_hat, 1, cfg.dx)
    du_true_dx = derivative(u_true, 2, cfg.dx)
    du_true_dy = derivative(u_true, 1, cfg.dx)
    dv_true_dx = derivative(v_true, 2, cfg.dx)
    dv_true_dy = derivative(v_true, 1, cfg.dx)

    kinetic_hat = 0.5 * jnp.mean(u_hat**2 + v_hat**2, axis=(1, 2))
    kinetic_true = 0.5 * jnp.mean(u_true**2 + v_true**2, axis=(1, 2))
    energy = jnp.mean(
        ((kinetic_hat - kinetic_true) / (kinetic_true + max(cfg.energy_floor, 1e-12)))
        ** 2
    )
    sobolev = (
        jnp.mean(
            (du_hat_dx - du_true_dx) ** 2
            + (du_hat_dy - du_true_dy) ** 2
            + (dv_hat_dx - dv_true_dx) ** 2
            + (dv_hat_dy - dv_true_dy) ** 2
        )
        / cfg.gradient_ref
    )
    divergence = (
        jnp.mean(((du_hat_dx + dv_hat_dy) - (du_true_dx + dv_true_dy)) ** 2)
        / cfg.gradient_ref
    )
    return {"energy": energy, "sobolev": sobolev, "divergence": divergence}
