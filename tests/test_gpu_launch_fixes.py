"""Regression gates for the final pre-submission feedback."""

import json
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from fluidgenie.training.initialization import initialization_rngs


@pytest.mark.parametrize("kind", ["lam", "tokenizer", "st_dynamics", "base_dynamics"])
def test_stochastic_training_initialization_has_independent_dropout_rng(kind):
    rngs = initialization_rngs(jax.random.PRNGKey(4))
    assert not np.array_equal(rngs["params"], rngs["dropout"])
    video = jnp.ones((1, 3, 4, 4, 3), dtype=jnp.float32)
    if kind == "lam":
        from fluidgenie.models.lam import LatentActionModel

        model = LatentActionModel(3, 8, 4, 8, 2, 1, 2, 0.1, 0.1)
        output = model.init(rngs, {"videos": video}, training=True)
    elif kind == "tokenizer":
        from fluidgenie.models.tokenizer_st import TokenizerSTVQVAE

        model = TokenizerSTVQVAE(3, 8, 4, 8, 2, 1, 2, 0.1, 0.1)
        output = model.init(rngs, {"videos": video}, training=True)
    elif kind == "st_dynamics":
        from fluidgenie.models.dynamics_st import DynamicsSTMaskGIT

        model = DynamicsSTMaskGIT(8, 8, 1, 2, 0.1, 0.0, 1.0, 0.2)
        batch = {
            "video_tokens": jnp.zeros((1, 3, 2, 2), dtype=jnp.int32),
            "mask_rng": jax.random.PRNGKey(8),
        }
        output = model.init(rngs, batch, training=True)
    else:
        from fluidgenie.models.base_dynamics import TransformerDynamics, DynConfig

        model = TransformerDynamics(
            DynConfig(
                vocab_size=9, d_model=8, n_heads=2, n_layers=1, dropout=0.1, max_len=8
            )
        )
        output = model.init(rngs, jnp.zeros((1, 6), dtype=jnp.int32), train=True)
    assert output["params"] and all(
        np.isfinite(x).all() for x in jax.tree.leaves(output["params"])
    )


def test_code_change_fraction_detects_discrete_collapse_despite_float_roundoff():
    from scripts.evaluate_lam_discovery import discrete_code_change_fraction

    codes = np.zeros(12, dtype=int)
    permutation = np.arange(12)[::-1]
    assert discrete_code_change_fraction(codes, permutation) == 0.0
    assert discrete_code_change_fraction(np.arange(12), permutation) == 1.0


def test_nonfinite_motion_cannot_pass_motion_or_stationarity(tmp_path):
    from fluidgenie.data.quality import validate

    fields = np.zeros((8, 4, 4, 3), dtype=np.float32)
    fields[4:] = np.inf
    np.save(tmp_path / "episode_000000.npy", fields)
    with np.errstate(invalid="ignore", over="ignore"):
        result = validate(tmp_path)
    assert (
        not result["finite_gate"]
        and not result["persistent_motion_gate"]
        and not result["stationarity_gate"]
    )


def test_stepper_separates_force_and_density_signatures():
    from fluidgenie.data.gen_phiflow_ns2d import _cached_stepper, _PHYSICS_FIELDS
    from configs.gen_data_configs import NS2DConfig

    cfg = NS2DConfig()
    key = tuple(getattr(cfg, k) for k in _PHYSICS_FIELDS)
    a = _cached_stepper(key, False, True, True)
    assert a is _cached_stepper(key, False, True, True)
    assert a is not _cached_stepper(key, False, False, True)
    assert a is not _cached_stepper(key, False, True, False)


def test_null_architecture_metadata_fails_clearly(tmp_path):
    from configs.config_io import save_config_json
    from configs.eval_configs import RolloutConfig
    from fluidgenie.eval.loaders import load_dynamics_model, load_lam_model

    save_config_json(tmp_path, {"config": {"architecture_version": None}})
    cfg = RolloutConfig(
        "unused",
        "unused",
        str(tmp_path),
        "unused",
        model_type="st_maskgit",
        use_lam=True,
        lam_ckpt=str(tmp_path),
    )
    for load in [
        lambda: load_dynamics_model(cfg, 12, jax.random.PRNGKey(0)),
        lambda: load_lam_model(cfg, 4, 4, 3, jax.random.PRNGKey(0)),
    ]:
        with pytest.raises(ValueError, match="architecture_version"):
            load()
