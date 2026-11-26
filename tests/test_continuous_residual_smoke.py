import jax
import jax.numpy as jnp

from fluidgenie.models.continuous_residual import ContinuousResidualUNet, bounded_residual_delta
from fluidgenie.training.train_continuous_residual import residual_unroll_loss
from fluidgenie.training.continuous_residual_physics_losses import PhysicsLossConfig


def test_continuous_residual_shapes_and_unroll_loss():
    context = jnp.zeros((2, 2, 16, 16, 3), dtype=jnp.float32)
    targets = jnp.ones((2, 4, 16, 16, 3), dtype=jnp.float32)
    model = ContinuousResidualUNet(channels=3, hidden_dims=(8, 16, 32), num_res_blocks=1)
    params = model.init(jax.random.PRNGKey(0), context, training=True)["params"]
    delta = model.apply({"params": params}, context, training=False)
    assert delta.shape == targets[:, 0].shape
    loss, _ = residual_unroll_loss(model.apply, params, context, targets)
    assert jnp.isfinite(loss)


def test_bounded_residual_is_linear_near_zero_and_caps_extremes():
    scales = (0.69, 0.65, 0.50)
    typical = jnp.asarray([[[[0.01, -0.01, 0.02]]]], dtype=jnp.float32)
    bounded_typical = bounded_residual_delta(typical, scales)
    assert jnp.max(jnp.abs(bounded_typical - typical)) < 2e-5

    extreme = jnp.asarray([[[[1e6, -1e6, 1e6]]]], dtype=jnp.float32)
    bounded_extreme = bounded_residual_delta(extreme, scales)
    assert jnp.all(jnp.abs(bounded_extreme) <= jnp.asarray(scales)[None, None, None, :])


def test_zero_context_noise_reproduces_unperturbed_loss_and_noise_is_finite():
    context = jnp.ones((1, 2, 16, 16, 3), dtype=jnp.float32) * 0.1
    targets = jnp.zeros((1, 4, 16, 16, 3), dtype=jnp.float32)
    model = ContinuousResidualUNet(channels=3, hidden_dims=(8, 16, 32), num_res_blocks=1)
    params = model.init(jax.random.PRNGKey(3), context, training=True)["params"]
    clean, _ = residual_unroll_loss(model.apply, params, context, targets)
    zero, _ = residual_unroll_loss(model.apply, params, context, targets, jax.random.PRNGKey(4), (0.0, 0.0, 0.0))
    noisy, _ = residual_unroll_loss(model.apply, params, context, targets, jax.random.PRNGKey(4), (0.01, 0.01, 0.01))
    assert jnp.allclose(clean, zero)
    assert jnp.isfinite(noisy)


def test_physics_endpoint_losses_are_finite_and_zero_weights_preserve_data_loss():
    context = jnp.ones((1, 2, 16, 16, 3), dtype=jnp.float32) * 0.1
    targets = jnp.zeros((1, 4, 16, 16, 3), dtype=jnp.float32)
    model = ContinuousResidualUNet(channels=3, hidden_dims=(8, 16, 32), num_res_blocks=1)
    params = model.init(jax.random.PRNGKey(5), context, training=True)["params"]
    clean, _ = residual_unroll_loss(model.apply, params, context, targets)
    physics = PhysicsLossConfig(energy_weight=0.1, dx=1 / 16, energy_floor=1e-3, gradient_ref=1.0)
    total, metrics = residual_unroll_loss(
        model.apply, params, context, targets, physics=physics,
        minimum=jnp.asarray([-1.0, -1.0, 0.0]), span=jnp.asarray([2.0, 2.0, 1.0]),
    )
    assert jnp.isfinite(total)
    assert jnp.isfinite(metrics["energy_loss"])
    assert jnp.allclose(metrics["data_loss"], clean)
