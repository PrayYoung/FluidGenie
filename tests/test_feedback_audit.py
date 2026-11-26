"""Regression checks for confirmed findings in the second branch review."""

import json
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.training.train_state import TrainState
from fluidgenie.eval.samplers.maskgit import (
    maskgit_rollout_tokens,
    st_maskgit_rollout_tokens,
)
from fluidgenie.training.codebook_restart import restart_codes
from fluidgenie.training.optimizers import make_optimizer
from fluidgenie.eval.metrics import fluid_physics_metrics
from fluidgenie.training.run_metadata import prepare_run
from scripts.evaluate_lam_discovery import discovery_windows


def test_maskgit_preserves_commits_against_hostile_unmasked_logits():
    class ST:
        def apply(self, variables, batch, training=False):
            mask = batch["mask"][:, -1].reshape(1, 16)
            # Masked positions predict 1. Unmasked positions adversarially predict 2.
            preferred = jnp.where(mask, 1, 2)
            logits = jax.nn.one_hot(preferred, 3) * 10
            return {"token_logits": logits[:, None]}

    ctx = jnp.zeros((1, 2, 4, 4), dtype=jnp.int32)
    bg = jnp.zeros((1, 4, 4), dtype=bool).at[:, :3].set(True)
    for background in [None, bg]:
        out = jax.jit(
            lambda key: st_maskgit_rollout_tokens(
                ST(), {}, ctx, 8, key, bg_mask=background
            )
        )(jax.random.PRNGKey(3))
        assert (
            np.all(np.asarray(out)[~np.asarray(background)] == 1)
            if background is not None
            else np.all(out == 1)
        )
        if background is not None:
            assert np.all(np.asarray(out)[np.asarray(background)] == 0)

    class Base:
        def apply(self, variables, seq, **kwargs):
            preferred = jnp.where(seq == 3, 1, 2)
            return (jax.nn.one_hot(preferred, 4) * 10).at[..., 3].set(100)

    out = maskgit_rollout_tokens(
        Base(), {}, jnp.zeros((1, 2), dtype=jnp.int32), 16, 4, 3, 8
    )
    assert np.all(out == 1)  # Never emits mask ID, never overwrites committed 1 with 2.


def test_discovery_windows_cover_each_transition_once_at_trained_length():
    episode = np.arange(16)[:, None, None, None]
    transitions = []
    for start, window, selection in discovery_windows(episode, 4):
        assert len(window) == 4
        transitions.extend(np.arange(start, start + 3)[selection].tolist())
    assert transitions == list(range(15))
    with pytest.raises(ValueError):
        list(discovery_windows(episode[:2], 4))


def test_restart_codes_distinguishes_duplicate_candidates_with_chained_optimizer():
    state = TrainState.create(
        apply_fn=lambda *_: None,
        params={"vq": {"codebook": jnp.zeros((8, 4))}},
        tx=make_optimizer(1e-3, 20),
    )
    state, count = restart_codes(
        state, jnp.zeros(8), jnp.ones((2, 4)), jax.random.PRNGKey(1)
    )
    values = np.asarray(state.params["vq"]["codebook"])
    assert count == 8 and np.unique(values, axis=0).shape[0] == 8
    assert np.allclose(np.linalg.norm(values, axis=1), 1)


def test_persistence_reference_is_frozen_context_and_lag1_is_separate():
    gt = np.broadcast_to(np.arange(1, 5)[:, None, None, None], (4, 4, 4, 3)).copy()
    out = fluid_physics_metrics(gt, np.zeros_like(gt), last_context=np.zeros((4, 4, 3)))
    assert np.allclose(out["persistence_mse"], [1, 4, 9, 16])
    assert np.allclose(out["lag1_persistence_mse"], 1)
    assert np.allclose(out["lag1_step_mse"], 1)
    assert out["persistence_reference"] == "last_context"


def test_prepare_run_embeds_portable_manifest(tmp_path):
    @dataclass
    class Config:
        data: str

    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "episode_000000.npy", np.zeros((4, 4, 4, 3)))
    (data / "episode_000000.json").write_text(json.dumps({"seed": 1}))
    out = prepare_run(tmp_path / "run", "test", Config(str(data)))
    saved = json.loads((out / "config.json").read_text())
    assert saved["implementation_version"] == 2
    assert len(saved["config"]["training_manifest"][0]["sha256"]) == 64


def test_action_batches_reject_ambiguous_temporal_alignment(tmp_path):
    from fluidgenie.data.dataset_npz import NPZSequenceDataset

    with pytest.raises(ValueError, match="pred=1"):
        NPZSequenceDataset(str(tmp_path), action_dim=5, pred=2)


def test_periodic_gradients_cross_boundary():
    from fluidgenie.training.losses import spatial_grads

    x = jnp.zeros((1, 8, 8, 3)).at[:, 0, 0].set(1)
    gradients = spatial_grads(x)
    assert any(
        np.any(np.asarray(g)[:, -1]) or np.any(np.asarray(g)[:, :, -1])
        for g in gradients
    )


@pytest.mark.parametrize("seq_len", [2, 4])
def test_lam_discovery_executes_long_two_channel_episodes_in_trained_windows(
    tmp_path, seq_len
):
    from dataclasses import asdict
    from configs.model_configs import LAMConfig
    from configs.config_io import save_config_json
    from fluidgenie.models.lam import LatentActionModel
    from fluidgenie.training.checkpoint_utils import save_params
    from scripts.evaluate_lam_discovery import Args, evaluate

    cfg = LAMConfig(
        data="unused",
        out="unused",
        seq_len=seq_len,
        model_dim=8,
        latent_dim=4,
        num_latents=8,
        patch_size=2,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
        codebook_dropout=0.0,
    )
    model = LatentActionModel(2, 8, 4, 8, 2, 1, 2, 0.0, 0.0)
    params = model.init(
        jax.random.PRNGKey(1), {"videos": jnp.zeros((1, seq_len, 4, 4, 2))}
    )["params"]
    checkpoint = tmp_path / "lam"
    save_config_json(checkpoint, {"kind": "lam", "config": asdict(cfg)})
    save_params(checkpoint, "latest", params)
    for seed, split in enumerate(["train", "test"]):
        data = tmp_path / split
        data.mkdir()
        (data / "actions").mkdir()
        np.save(
            data / "episode_000000.npy",
            np.random.default_rng(seed).normal(size=(16, 4, 4, 2)).astype(np.float32),
        )
        np.save(data / "actions/episode_000000.npy", np.arange(16) % 5)
        (data / "episode_000000.json").write_text(json.dumps({"seed": seed}))
    stats = tmp_path / "stats.npz"
    np.savez(stats, mean=np.zeros(2), std=np.ones(2))
    out = evaluate(
        Args(
            str(checkpoint / "latest"),
            str(tmp_path / "train"),
            str(tmp_path / "test"),
            str(stats),
            str(tmp_path / "result.json"),
            frames=16,
        )
    )
    assert sum(out["test_action_counts"]) == 15
    assert np.isfinite(out["conditioned_mse"])


def test_conditional_diagnostic_initialization_and_sampling(tmp_path):
    from configs.config_io import save_config_json
    from fluidgenie.models.flow_matching import ConditionalFlowUNet
    from fluidgenie.training.checkpoint_utils import save_params
    from fluidgenie.eval.flow_matching_diagnostics import _load_model
    from fluidgenie.eval.samplers.flow_matching import sample_flow_matching_heun

    cfg = dict(
        context=2,
        hidden_dims=[4, 8, 8],
        time_dim=8,
        num_res_blocks=1,
        modern=True,
        action_dim=5,
    )
    model = ConditionalFlowUNet(3, 2, (4, 8, 8), 8, 1, True, 5)
    context = jnp.zeros((1, 2, 4, 4, 3))
    action = jax.nn.one_hot(jnp.array([1]), 5)
    params = model.init(
        jax.random.PRNGKey(0), context[:, 0], jnp.zeros(1), context, action_cond=action
    )["params"]
    save_config_json(tmp_path, {"kind": "flow_matching_fm0", "config": cfg})
    save_params(tmp_path, "latest", params)
    loaded, restored = _load_model(str(tmp_path / "latest"), 4, 4, 3)
    result = sample_flow_matching_heun(
        loaded, restored, context, jax.random.PRNGKey(2), 2, action_cond=action
    )
    assert result.shape == (1, 4, 4, 3) and np.isfinite(result).all()


def test_conditional_rollout_uses_each_aligned_oracle_action(tmp_path, monkeypatch):
    import fluidgenie.eval.flow_matching_rollout as rollout
    from configs.config_io import save_config_json
    from fluidgenie.models.flow_matching import ConditionalFlowUNet
    from fluidgenie.training.checkpoint_utils import save_params

    train, test = tmp_path / "train", tmp_path / "test"
    for seed, directory in [(1, train), (2, test)]:
        directory.mkdir()
        np.save(
            directory / "episode_000000.npy",
            np.full((8, 4, 4, 3), seed, dtype=np.float32),
        )
        (directory / "episode_000000.json").write_text(json.dumps({"seed": seed}))
    (test / "actions").mkdir()
    np.save(test / "actions/episode_000000.npy", np.array([0, 1, 2, 3, 4, 0, 1, 2]))
    cfg = dict(
        context=2,
        hidden_dims=[4, 8, 8],
        time_dim=8,
        num_res_blocks=1,
        modern=True,
        action_dim=5,
        data=str(train),
    )
    model = ConditionalFlowUNet(3, 2, (4, 8, 8), 8, 1, True, 5)
    ctx = jnp.zeros((1, 2, 4, 4, 3))
    action = jnp.zeros((1, 5))
    params = model.init(
        jax.random.PRNGKey(0), ctx[:, 0], jnp.zeros(1), ctx, action_cond=action
    )["params"]
    checkpoint = tmp_path / "model"
    save_config_json(checkpoint, {"kind": "flow_matching_fm0", "config": cfg})
    save_params(checkpoint, "latest", params)

    def controlled(model, params, context, key, steps, action_cond=None):
        return (
            context[:, -1]
            + jnp.sum(action_cond * jnp.arange(5), axis=-1)[:, None, None, None]
        )

    monkeypatch.setattr(rollout, "sample_flow_matching_heun", controlled)
    out = tmp_path / "out"
    rollout.run_flow_matching_rollout(
        rollout.FlowMatchingRolloutConfig(
            str(test / "episode_000000.npy"),
            str(checkpoint / "latest"),
            str(out),
            horizon=3,
            make_visualizations=False,
        )
    )
    pred = np.load(out / "rollout_arrays.npz")["pred"]
    assert np.allclose(pred[:, 0, 0, 0], [3, 5, 8])


def test_drift_diagnostic_accepts_zscore_only_stats_and_reports_unknown_bounds(
    tmp_path,
):
    from configs.config_io import save_config_json
    from fluidgenie.models.continuous_residual import ContinuousResidualUNet
    from fluidgenie.training.checkpoint_utils import save_params
    from fluidgenie.eval.cr1_v2_drift_diagnosis import (
        DriftDiagnosisConfig,
        run_drift_diagnosis,
    )

    cfg = dict(
        context=2,
        hidden_dims=[4, 8, 8],
        num_res_blocks=1,
        residual_scales=[0.1, 0.1, 0.1],
    )
    model = ContinuousResidualUNet(3, 2, (4, 8, 8), 1, (0.1, 0.1, 0.1))
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2, 4, 4, 3)))["params"]
    checkpoint = tmp_path / "model"
    save_config_json(
        checkpoint, {"kind": "continuous_residual_cr1_v2_k4_bounded", "config": cfg}
    )
    save_params(checkpoint, "latest", params)
    np.savez(checkpoint / "stats.npz", mean=np.zeros(3), std=np.ones(3))
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "episode_000000.npy", np.zeros((6, 4, 4, 3), dtype=np.float32))
    result = run_drift_diagnosis(
        DriftDiagnosisConfig(
            data_dir=str(data),
            stats_path="",
            dyn_ckpt=str(checkpoint / "latest"),
            out_dir=str(tmp_path / "out"),
            episode_indices=(0,),
            horizon=3,
        )
    )
    assert result["training_bounds"]["density"] is None
    assert result["step_means"][1]["density_out_of_bounds"] is None
