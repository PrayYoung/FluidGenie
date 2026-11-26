"""Regression gates for the photographed review's reproducible implementation issues."""

import json
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from fluidgenie.data.splits import assert_disjoint
from fluidgenie.data.dataset_npz import NPZSequenceDataset
from fluidgenie.models.periodic import PeriodicConv
from fluidgenie.models.continuous_residual import ContinuousResidualUNet
from fluidgenie.models.flow_matching import ConditionalFlowUNet
from fluidgenie.models.lam import LatentActionModel
from fluidgenie.models.tokenizer_st import TokenizerSTVQVAE
from fluidgenie.training.train_continuous_residual import (
    residual_unroll_loss,
    make_physics_train_step,
    TrainState,
)
from fluidgenie.training.continuous_residual_physics_losses import (
    PhysicsLossConfig,
    periodic_derivative,
)
from fluidgenie.training.optimizers import make_optimizer
from fluidgenie.training.codebook_restart import restart_codes
from fluidgenie.eval.samplers.flow_matching import sample_flow_matching_heun
from fluidgenie.eval.metrics import fluid_physics_metrics
from fluidgenie.eval.action_discovery import discovery_metrics


def test_heun_never_queries_endpoint_even_for_single_step():
    class EndpointSensitive:
        def apply(self, variables, x, tau, context, training=False):
            return jnp.ones_like(x) / (1 - tau[:, None, None, None])

    ctx = jnp.zeros((1, 2, 4, 4, 3))
    for steps in (1, 2, 8):
        result = sample_flow_matching_heun(
            EndpointSensitive(), {}, ctx, jax.random.PRNGKey(0), steps
        )
        assert bool(jnp.isfinite(result).all())


def test_periodic_convolution_wraps_and_derivative_has_no_wall():
    x = jnp.zeros((1, 8, 8, 1)).at[0, 0, 0, 0].set(1)
    model = PeriodicConv(1)
    variables = model.init(jax.random.PRNGKey(0), x)
    variables["params"]["Conv_0"]["kernel"] = jnp.ones_like(
        variables["params"]["Conv_0"]["kernel"]
    )
    y = model.apply(variables, x)
    assert y[0, -1, -1, 0] == 1 and y[0, 4, 4, 0] == 0
    assert bool(jnp.all(periodic_derivative(jnp.ones((1, 8, 8)), 2, 1 / 8) == 0))


def test_split_rejects_reused_seed_and_reused_content():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for split, seed, value in [("train", 0, 0), ("heldout", 99991, 1)]:
            p = root / split
            p.mkdir()
            np.save(
                p / "episode_000000.npy", np.full((3, 4, 4, 3), value, dtype=np.float32)
            )
            (p / "episode_000000.json").write_text(json.dumps({"seed": seed}))
        assert assert_disjoint(root / "train", root / "heldout")["disjoint"]
        (root / "heldout/episode_000000.json").write_text(json.dumps({"seed": 0}))
        try:
            assert_disjoint(root / "train", root / "heldout")
        except ValueError:
            pass
        else:
            raise AssertionError("Seed overlap accepted")
        (root / "heldout/episode_000000.json").write_text(json.dumps({"seed": 99991}))
        np.save(
            root / "heldout/episode_000000.npy",
            np.zeros((3, 4, 4, 3), dtype=np.float32),
        )
        try:
            assert_disjoint(root / "train", root / "heldout")
        except ValueError:
            pass
        else:
            raise AssertionError("Content overlap accepted")


def test_action_alignment_and_training_only_zscore():
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp)
        x = np.ones((6, 4, 4, 3), dtype=np.float32) * 3
        np.save(p / "episode_000000.npy", x)
        np.save(p / "episode_000000.actions.npy", np.arange(6) % 5)
        np.savez(p / "stats.npz", mean=np.ones(3), std=np.ones(3) * 2)
        ds = NPZSequenceDataset(
            temp, context=2, stats_path=str(p / "stats.npz"), action_dim=5
        )
        context, target, action = ds[0]
        assert len(ds.files) == 1 and int(action.argmax()) == 1
        assert np.allclose(context, 1, atol=1e-6) and np.allclose(target, 1, atol=1e-6)


def test_stable_cr_starts_identity_and_has_finite_physics_gradients():
    ctx = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 8, 8, 3)) * 0.1
    target = jnp.repeat(ctx[:, -1:], 4, axis=1) + 0.01
    for seed in (0, 1, 2):
        model = ContinuousResidualUNet(
            3,
            hidden_dims=(4, 8, 16),
            num_res_blocks=1,
            residual_scales=(0.5,) * 3,
            stable=True,
        )
        params = model.init(jax.random.PRNGKey(seed), ctx)["params"]
        assert bool(jnp.all(model.apply({"params": params}, ctx) == 0))
        state = TrainState.create(
            apply_fn=model.apply, params=params, tx=make_optimizer(3e-4, 10, 2)
        )
        step = make_physics_train_step(
            PhysicsLossConfig(sobolev_weight=0.01, periodic=True),
            jnp.zeros(3),
            jnp.ones(3),
        )
        state, metrics = step(state, ctx, target)
        assert all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(state.params))
        assert bool(jnp.isfinite(metrics["grad_norm"]))


def test_lam_inference_cannot_read_later_frames():
    videos = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 8, 8, 3))
    m = LatentActionModel(3, 16, 8, 8, 4, 1, 2, 0.0, 0.0)
    p = m.init(jax.random.PRNGKey(1), {"videos": videos})
    a = m.apply(p, videos, training=False, method=m.vq_encode)
    b = m.apply(p, videos.at[:, 3].add(100), training=False, method=m.vq_encode)
    assert np.allclose(a["emb"][:2], b["emb"][:2], atol=1e-6)
    assert np.array_equal(a["indices"][:2], b["indices"][:2])


def test_joint_st_decoder_prefix_is_causal():
    videos = jax.random.normal(jax.random.PRNGKey(0), (1, 3, 8, 12, 3))
    m = TokenizerSTVQVAE(3, 16, 8, 8, 4, 1, 2, 0.0, 0.0)
    variables = m.init(jax.random.PRNGKey(0), {"videos": videos})
    # Make the final decoder projection nonzero so this is not a vacuous zero-head test.
    variables["params"]["decoder"]["Dense_1"]["kernel"] = (
        jax.random.normal(
            jax.random.PRNGKey(4),
            variables["params"]["decoder"]["Dense_1"]["kernel"].shape,
        )
        * 0.01
    )
    tokens = m.apply(variables, videos, training=False, method=m.encode_video)
    full = m.apply(variables, tokens, (8, 12), method=m.decode_video_tokens)
    prefix = m.apply(variables, tokens[:, :2], (8, 12), method=m.decode_video_tokens)
    assert np.allclose(full[:, :2], prefix, atol=1e-5)


def test_physics_spectrum_conserves_energy_and_persistence_is_explicit():
    n = 16
    y, x = np.mgrid[:n, :n] / n
    fields = np.zeros((3, n, n, 3))
    fields[..., 0] = np.sin(2 * np.pi * y)
    m = fluid_physics_metrics(fields, fields)
    assert np.allclose(
        m["prediction"]["energy_spectrum"].sum(axis=1),
        m["prediction"]["kinetic_energy"],
    )
    assert np.allclose(m["prediction"]["divergence_rms"], 0)
    assert np.allclose(m["vorticity_correlation"], 1)
    assert np.allclose(m["persistence_mse"], 0)


def test_discovery_requires_out_of_sample_purity_and_ablation_effect():
    actions = np.tile(np.arange(5), 10)
    m = discovery_metrics(actions, actions, actions, actions, 5, 0.09, 0.2, 0.2)
    assert m["discovery_gate_passed"] and np.isclose(m["nmi"], 1)
    m = discovery_metrics(
        actions, actions, actions, np.zeros_like(actions), 5, 0.2, 0.2, 0.2
    )
    assert not m["discovery_gate_passed"]


def test_modern_flow_action_path_and_gradients_are_finite():
    from fluidgenie.training.train_flow_matching import flow_matching_loss

    m = ConditionalFlowUNet(
        3,
        hidden_dims=(4, 8, 16),
        time_dim=16,
        num_res_blocks=1,
        modern=True,
        action_dim=5,
    )
    context = jnp.zeros((1, 2, 8, 8, 3))
    action = jax.nn.one_hot(jnp.array([3]), 5)
    params = m.init(
        jax.random.PRNGKey(0), context[:, 0], jnp.zeros(1), context, action_cond=action
    )["params"]
    loss, grad = jax.value_and_grad(
        lambda p: flow_matching_loss(
            m.apply,
            p,
            context,
            context[:, 0],
            jax.random.PRNGKey(3),
            action,
            (0.01, 0.01),
            jnp.ones(3),
        )[0]
    )(params)
    assert jnp.isfinite(loss) and all(
        bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(grad)
    )


def test_flow_stays_float32_when_simulator_enables_jax_x64():
    with jax.experimental.enable_x64():
        context = jnp.zeros((1, 2, 8, 8, 3), dtype=jnp.float32)
        action = jax.nn.one_hot(jnp.array([3]), 5, dtype=jnp.float32)
        model = ConditionalFlowUNet(
            3,
            hidden_dims=(4, 8, 16),
            time_dim=16,
            num_res_blocks=1,
            modern=True,
            action_dim=5,
        )
        params = model.init(
            jax.random.PRNGKey(0),
            context[:, 0],
            jnp.zeros(1, dtype=jnp.float32),
            context,
            action_cond=action,
        )["params"]
        result = sample_flow_matching_heun(
            model, params, context, jax.random.PRNGKey(1), 2, action
        )
        assert result.dtype == jnp.float32 and bool(jnp.isfinite(result).all())
