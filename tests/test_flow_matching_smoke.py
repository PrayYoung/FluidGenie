import jax
import jax.numpy as jnp

from fluidgenie.eval.samplers.flow_matching import (
    sample_flow_matching_euler,
    sample_flow_matching_heun,
)
from fluidgenie.models.flow_matching import ConditionalFlowUNet
from fluidgenie.training.train_flow_matching import flow_matching_loss


def test_fm0_shapes_loss_and_ode_samplers():
    context = jnp.zeros((2, 2, 16, 16, 3), dtype=jnp.float32)
    target = jnp.ones((2, 16, 16, 3), dtype=jnp.float32)
    model = ConditionalFlowUNet(channels=3, hidden_dims=(8, 16, 32), time_dim=16)
    key = jax.random.PRNGKey(0)
    params = model.init(key, target, jnp.zeros((2,)), context, training=True)["params"]
    loss, _ = flow_matching_loss(model.apply, params, context, target, jax.random.PRNGKey(1))
    assert jnp.isfinite(loss)
    for sampler in (sample_flow_matching_euler, sample_flow_matching_heun):
        sample = sampler(model, params, context, jax.random.PRNGKey(2), ode_steps=2)
        assert sample.shape == target.shape
        assert jnp.all(jnp.isfinite(sample))
