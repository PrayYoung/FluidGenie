"""The CUDA gate must exercise the same compiled math as the trainer."""

import jax
import jax.numpy as jnp
import numpy as np
from fluidgenie.models.continuous_residual import ContinuousResidualUNet
from fluidgenie.training.train_continuous_residual import residual_unroll_loss
from scripts.continuous_residual_gpu_smoke import make_gradient_gate


def test_jit_gate_preserves_eager_loss_and_gradients_on_small_cpu_fixture():
    model = ContinuousResidualUNet(3, 2, (4, 8, 8), 2, (0.5, 0.5, 0.5), True)
    context = jax.random.normal(
        jax.random.PRNGKey(0), (1, 2, 4, 4, 3), dtype=jnp.float32
    )
    targets = jax.random.normal(
        jax.random.PRNGKey(1), (1, 2, 4, 4, 3), dtype=jnp.float32
    )
    params = model.init(jax.random.PRNGKey(2), context)["params"]
    objective = lambda p: residual_unroll_loss(model.apply, p, context, targets)[0]
    expected = jax.value_and_grad(objective)(params)
    gate = make_gradient_gate(model.apply)
    actual = gate(params, context, targets)
    assert np.allclose(actual[0], expected[0], rtol=1e-5, atol=1e-6)
    for compiled, eager in zip(
        jax.tree.leaves(actual[1]), jax.tree.leaves(expected[1])
    ):
        assert np.isfinite(compiled).all() and np.allclose(
            compiled, eager, rtol=1e-4, atol=1e-5
        )
    # Dynamic inputs must remain inputs rather than constants captured in compilation.
    changed = gate(params, context, targets * 2)
    assert not np.isclose(changed[0], actual[0])
