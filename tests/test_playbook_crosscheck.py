"""Checks for gaps revealed by the detailed photographed playbook."""

import json
import tempfile
from pathlib import Path
from dataclasses import fields, replace
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from configs.gen_data_configs import GenDataArgs, NS2DConfig
from fluidgenie.data.gen_phiflow_ns2d import (
    config_from_args,
    _cached_stepper,
    localized_jet_kernel,
)
from fluidgenie.models.st_transformer import VectorQuantizer
from fluidgenie.models.lam import LatentActionModel
from fluidgenie.eval.metrics import fluid_physics_metrics
from fluidgenie.eval.samplers.maskgit import st_maskgit_rollout_tokens
from fluidgenie.training.checkpoint_utils import (
    save_training_state,
    restore_training_state,
)
from fluidgenie.training.train_lam import train_step
from fluidgenie.training.codebook_restart import codebook_health


def test_all_simulator_cli_fields_are_forwarded_and_step_cache_is_seed_independent():
    args = GenDataArgs(
        init_velocity_scale=0.25, forcing_smoothness=2.5, implicit=True, linear_drag=0.2
    )
    cfg = config_from_args(args)
    assert (
        cfg.init_velocity_scale == 0.25
        and cfg.forcing_smoothness == 2.5
        and cfg.use_implicit_diffusion
        and cfg.linear_drag == 0.2
    )
    from fluidgenie.data.gen_phiflow_ns2d import _PHYSICS_FIELDS

    key = tuple(getattr(cfg, k) for k in _PHYSICS_FIELDS)
    assert _cached_stepper(key, True) is _cached_stepper(key, True)
    assert abs(localized_jet_kernel(64, 0.14).mean()) < 1e-7


def test_reserved_background_code_cannot_be_assigned_to_foreground_even_with_dropout():
    model = VectorQuantizer(4, 8, 1.0, reserve_code0=True)
    x = jnp.ones((12, 4), dtype=jnp.float32)
    params = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}, x, True
    )
    result = model.apply(params, x, True, rngs={"dropout": jax.random.PRNGKey(2)})
    assert bool(jnp.all(result[-1] != 0))


def test_physics_ratios_times_and_two_persistence_references():
    n = 16
    y, x = np.mgrid[:n, :n] / n
    gt = np.zeros((4, n, n, 3))
    gt[..., 0] = np.sin(2 * np.pi * y)
    gt[..., 2] = np.sin(2 * np.pi * x)
    metrics = fluid_physics_metrics(gt, gt, dt=0.02, substeps=4, last_context=gt[0])
    assert np.allclose(metrics["tke_ratio"], 1) and np.allclose(
        metrics["enstrophy_ratio"], 1
    )
    assert np.allclose(metrics["smoke_sharpness_ratio"], 1) and np.allclose(
        metrics["physical_times"], [0.08, 0.16, 0.24, 0.32]
    )
    assert metrics["vorticity_crossing_time_below_0_8"] is None
    assert np.allclose(metrics["frozen_context_persistence_mse"], 0)
    assert np.allclose(metrics["lag1_persistence_mse"], 0)
    assert np.allclose(
        metrics["mse_skill_vs_persistence"], 0
    )  # No skill can be claimed over an exact baseline.


def test_lam_257_token_prefix_causality_and_code_health_training():
    # 16x16 spatial patches plus one action prefix token per frame.
    video = jax.random.normal(
        jax.random.PRNGKey(0), (1, 3, 64, 64, 3), dtype=jnp.float32
    )
    model = LatentActionModel(3, 16, 8, 8, 4, 1, 2, 0.0, 0.0)
    params = model.init(jax.random.PRNGKey(1), {"videos": video})["params"]
    a = model.apply({"params": params}, video, False, method=model.vq_encode)
    b = model.apply(
        {"params": params}, video.at[:, 2].add(5), False, method=model.vq_encode
    )
    assert np.allclose(a["emb"][0], b["emb"][0], atol=1e-5)
    small = video[:, :, :8, :8]
    # Projected Fourier input width depends on the patch grid. Initialize the
    # tiny training grid independently from the full-grid causality check.
    small_params = model.init(jax.random.PRNGKey(1), {"videos": small})["params"]
    state = TrainState.create(apply_fn=model.apply, params=small_params, tx=optax.adam(1e-4))
    state, metrics = train_step(state, small, 0.25, jax.random.PRNGKey(2))
    assert "code_usage" in metrics and bool(
        jnp.isfinite(codebook_health(metrics["code_usage"])["code_perplexity"])
    )


def test_maskgit_sampler_runs_under_jit_and_scan_with_dynamic_ratio():
    class Mock:
        def apply(self, variables, batch, training=False):
            video = batch["video_tokens"]
            b, t, h, w = video.shape
            return {
                "token_logits": jnp.broadcast_to(
                    jnp.arange(4, dtype=jnp.float32), (b, t, h * w, 4)
                )
            }

    def rollout(ratio):
        def step(context, key):
            nxt = st_maskgit_rollout_tokens(
                Mock(), {}, context, 2, key, init_mask_ratio=ratio
            )
            return jnp.concatenate([context[:, 1:], nxt[:, None]], axis=1), nxt

        return jax.lax.scan(
            step,
            jnp.zeros((1, 2, 2, 2), dtype=jnp.int32),
            jax.random.split(jax.random.PRNGKey(0), 3),
        )[1]

    result = jax.jit(rollout)(jnp.asarray(1.0, dtype=jnp.float32))
    assert result.shape == (3, 1, 2, 2) and bool(jnp.all(result == 3))


def test_resume_restores_optimizer_rng_and_replays_data_cursor():
    state = TrainState.create(
        apply_fn=lambda *_: None, params={"w": jnp.ones(2)}, tx=optax.adam(1e-3)
    )
    state = state.apply_gradients(grads={"w": jnp.asarray([1.0, 2.0])})
    key = jax.random.PRNGKey(17)
    with tempfile.TemporaryDirectory() as temp:
        save_training_state(temp, state, key)
        fresh = TrainState.create(
            apply_fn=state.apply_fn, params={"w": jnp.zeros(2)}, tx=optax.adam(1e-3)
        )
        iterator = iter(range(10))
        restored, rng = restore_training_state(
            temp, fresh, jax.random.PRNGKey(0), iterator
        )
        assert (
            int(restored.step) == 1 and next(iterator) == 1 and np.array_equal(key, rng)
        )
        expected = state.apply_gradients(grads={"w": jnp.asarray([3.0, 4.0])})
        actual = restored.apply_gradients(grads={"w": jnp.asarray([3.0, 4.0])})
        assert np.array_equal(actual.params["w"], expected.params["w"])


def test_frozen_lam_flow_training_uses_video_without_action_label_files():
    """Positive wiring fixture only; its synthetic certificate is not research evidence."""
    import subprocess
    import sys
    from dataclasses import asdict
    from configs.model_configs import LAMConfig
    from configs.config_io import save_config_json
    from fluidgenie.training.checkpoint_utils import save_params
    from fluidgenie.cli.steering_demo import SteeringSession

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        data = root / "train"
        data.mkdir()
        video = (
            np.random.default_rng(0).normal(size=(6, 8, 8, 3)).astype(np.float32) * 0.1
        )
        episode = data / "episode_000000.npy"
        np.save(episode, video)
        stats = root / "stats.npz"
        np.savez(
            stats, mean=np.zeros(3, dtype=np.float32), std=np.ones(3, dtype=np.float32)
        )
        lam_config = LAMConfig(
            data=str(data),
            out=str(root / "lam"),
            stats=str(stats),
            model_dim=8,
            latent_dim=8,
            num_latents=8,
            patch_size=4,
            num_blocks=1,
            num_heads=2,
        )
        lam = LatentActionModel(3, 8, 8, 8, 4, 1, 2, 0.0, 0.0)
        params = lam.init(
            jax.random.PRNGKey(0), {"videos": jnp.asarray(video[None, :3])}
        )["params"]
        save_params(root / "lam", "latest", params)
        save_config_json(root / "lam", {"kind": "lam", "config": asdict(lam_config)})
        certificate = root / "synthetic_wiring_certificate.json"
        certificate.write_text(
            json.dumps(
                {
                    "fixture_only": True,
                    "gate": {"version": 4},
                    "shuffle_protocol": "episode_wide_transition_permutation_v4",
                    "discovery_gate_passed": True,
                    "five_action_mapping_gate": True,
                    "calibrated_code_by_action": [0, 1, 2, 3, 4],
                    "protocol": {
                        "checkpoint": str(root / "lam/latest"),
                        "calibration_dir": str(data),
                    },
                }
            )
        )
        command = [
            sys.executable,
            "-m",
            "fluidgenie.training.train_flow_matching",
            "--data",
            str(data),
            "--stats",
            str(stats),
            "--out",
            str(root / "flow"),
            "--steps",
            "2",
            "--batch",
            "1",
            "--hidden-dims",
            "4",
            "8",
            "16",
            "--time-dim",
            "16",
            "--num-res-blocks",
            "1",
            "--grain-workers",
            "0",
            "--tb",
            "0",
            "--checkpoint-every",
            "0",
            "--use-lam",
            "--lam-ckpt",
            str(root / "lam/latest"),
            "--lam-discovery",
            str(certificate),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr[-2000:]
        saved = json.loads((root / "flow/config.json").read_text())
        assert saved["implementation_version"] == 2
        assert saved["config"]["training_manifest"]
        assert saved["config"]["use_lam"] and saved["config"]["action_dim"] == 8
        assert not list(data.glob("*.actions.npy")) and not (data / "actions").exists()
        session = SteeringSession(
            str(root / "flow/latest"), str(episode), str(stats), ode_steps=1
        )
        prediction = session.step(3)
        assert prediction.shape == (8, 8, 3) and np.isfinite(prediction).all()
