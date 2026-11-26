import tempfile
from pathlib import Path
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from fluidgenie.training.codebook_restart import restart_codes
from fluidgenie.training.run_metadata import prepare_run


def test_dead_code_restart_preserves_used_codes_and_clears_only_dead_momentum():
    params = {"vq": {"codebook": jnp.eye(3)}}
    state = TrainState.create(
        apply_fn=lambda *_: None, params=params, tx=optax.adam(0.001)
    )
    state = state.apply_gradients(grads=jax.tree.map(jnp.ones_like, params))
    before = np.asarray(state.params["vq"]["codebook"]).copy()
    updated, count = restart_codes(
        state,
        jnp.array([2, 0, 4]),
        jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        jax.random.PRNGKey(0),
    )
    assert int(count) == 1
    assert np.array_equal(
        np.asarray(updated.params["vq"]["codebook"])[[0, 2]], before[[0, 2]]
    )
    assert np.isclose(np.linalg.norm(updated.params["vq"]["codebook"][1]), 1)
    assert np.allclose(updated.opt_state[0].mu["vq"]["codebook"][1], 0)
    assert np.allclose(
        updated.opt_state[0].mu["vq"]["codebook"][0],
        state.opt_state[0].mu["vq"]["codebook"][0],
    )


def test_fresh_run_metadata_prevents_mixed_or_overwritten_experiments():
    @dataclass
    class Config:
        seed: int = 1

    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp) / "run"
        prepare_run(out, "test", Config())
        assert (out / "config.json").exists()
        try:
            prepare_run(out, "test", Config())
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing experiment was overwritten")
