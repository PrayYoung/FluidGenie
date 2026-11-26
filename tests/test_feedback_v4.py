"""Behavioral regressions for the follow-up audit of commit 7a9a073."""

from dataclasses import dataclass
from io import BytesIO
import json
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax.training.train_state import TrainState
from scripts.evaluate_lam_discovery import episode_wide_shuffle, discovery_windows
from fluidgenie.eval.metrics import (
    fluid_physics_metrics,
    jsonable,
    steering_response_metrics,
)
from configs.config_io import save_config_json


@pytest.mark.parametrize("seq_len", [2, 4])
def test_episode_shuffle_moves_donors_across_short_training_windows(seq_len):
    episode = np.zeros((20, 2, 2, 2))
    actions = jnp.arange(19, dtype=jnp.float32)[:, None, None]
    shuffled, permutation = episode_wide_shuffle(actions, jax.random.PRNGKey(22))
    assert np.mean(np.asarray(permutation) != np.arange(19)) > 0.7
    assert np.array_equal(np.sort(np.asarray(shuffled).ravel()), np.arange(19))
    for start, _, _ in discovery_windows(episode, seq_len):
        assert np.array_equal(
            shuffled[start : start + seq_len - 1, 0, 0],
            np.asarray(permutation)[start : start + seq_len - 1],
        )


def test_undefined_physics_is_strict_json_null_and_fallback_omits_first_target():
    gt = np.broadcast_to(np.arange(1, 5)[:, None, None, None], (4, 4, 4, 3)).copy()
    result = fluid_physics_metrics(gt, gt + 1)
    assert np.array_equal(result["persistence_target_indices"], [1, 2, 3])
    assert result["persistence_reference"] == "first_target"
    assert np.isfinite(result["mse_skill_vs_persistence"]).all()
    undefined = fluid_physics_metrics(
        np.zeros_like(gt), np.ones_like(gt), last_context=np.zeros((4, 4, 3))
    )
    payload = jsonable(undefined)
    assert payload["mse_skill_vs_persistence"] == [None] * 4
    json.dumps(payload, allow_nan=False)
    assert jsonable({"a": np.array([np.inf, np.nan, -np.inf]), "b": float("inf")}) == {
        "a": [None, None, None],
        "b": None,
    }


def test_switching_joystick_can_report_nonstationarity_without_failing_required_gates(
    tmp_path,
):
    from scripts.validate_forced_data import Args, main

    values = np.broadcast_to(
        (np.arange(12) + 1)[:, None, None, None], (12, 4, 4, 3)
    ).astype(np.float32)
    np.save(tmp_path / "episode_000000.npy", values)
    args = Args(
        str(tmp_path), str(tmp_path / "quality.json"), require_stationarity=False
    )
    result = main(args)
    assert (
        result["finite_gate"]
        and result["persistent_motion_gate"]
        and not result["stationarity_gate"]
    )
    args.require_stationarity = True
    with pytest.raises(SystemExit, match="stationarity_gate"):
        main(args)


def test_steering_response_detects_opposite_zero_mean_jets():
    gt = np.zeros((3, 8, 8, 3))
    gt[..., 0] = np.sin(np.arange(8) * 2 * np.pi / 8)[None, None, :]
    gt[..., 2] = np.arange(8)[None, None, :]
    zero = np.zeros_like(gt)
    assert abs(gt[..., 0].mean()) < 1e-12
    match = steering_response_metrics(gt, gt, zero, zero)
    assert np.isclose(match["velocity_delta_cosine"], 1) and np.isclose(
        match["velocity_delta_cosine_localized"], 1
    )
    reverse = gt.copy()
    reverse[..., :2] *= -1
    mismatch = steering_response_metrics(reverse, gt, zero, zero)
    assert np.isclose(mismatch["velocity_delta_cosine"], -1) and np.isclose(
        mismatch["density_correlation"], 1
    )
    assert (
        steering_response_metrics(zero, zero, zero, zero)["velocity_delta_cosine"]
        is None
    )


def test_restart_handles_empty_candidates_float64_and_nonzero_adam_slots():
    from fluidgenie.training.codebook_restart import restart_codes
    from fluidgenie.training.optimizers import make_optimizer

    params = {"vq": {"codebook": jnp.eye(4, dtype=jnp.float32)}}
    state = TrainState.create(
        apply_fn=lambda *_: None, params=params, tx=make_optimizer(1e-3, 20)
    )
    state = state.apply_gradients(grads=jax.tree.map(jnp.ones_like, params))
    empty, count = restart_codes(
        state, jnp.zeros(4), jnp.empty((0, 4)), jax.random.PRNGKey(0)
    )
    assert empty is state and int(count) == 0
    with jax.experimental.enable_x64():
        updated, count = restart_codes(
            state,
            jnp.array([1, 0, 0, 0]),
            jnp.ones((2, 4), dtype=jnp.float64),
            jax.random.PRNGKey(1),
        )
    assert updated.params["vq"]["codebook"].dtype == jnp.float32 and int(count) == 3
    slots = []
    for path, value in jax.tree_util.tree_flatten_with_path(updated.opt_state)[0]:
        if any(getattr(k, "key", None) == "codebook" for k in path) and getattr(
            value, "shape", None
        ) == (4, 4):
            slots.append(np.asarray(value))
    assert len(slots) == 2
    assert all(np.allclose(x[1:], 0) and np.any(x[0] != 0) for x in slots)


def test_resume_rejects_kind_and_new_config_keys_but_allows_worker_change(tmp_path):
    from fluidgenie.training.run_metadata import prepare_run

    @dataclass
    class Config:
        steps: int = 3
        grain_workers: int = 0

    out = prepare_run(tmp_path / "run", "one", Config())
    (out / "train_state").mkdir()
    prepare_run(out, "one", Config(grain_workers=4), resume=True)
    with pytest.raises(ValueError, match="kind"):
        prepare_run(out, "two", Config(), resume=True)

    @dataclass
    class AddedConfig:
        steps: int = 3
        grain_workers: int = 0
        new_option: int = 1

    with pytest.raises(ValueError, match="new_option"):
        prepare_run(out, "one", AddedConfig(), resume=True)


def test_demo_inherits_nested_lam_settings_from_dynamics(tmp_path):
    from configs.eval_configs import DemoArgs, RolloutArgs, apply_ckpt_config_to_demo

    lam = tmp_path / "lam"
    dyn = tmp_path / "dyn"
    save_config_json(
        lam, {"config": {"model_dim": 32, "latent_dim": 8, "num_latents": 6}}
    )
    save_config_json(
        dyn,
        {
            "config": {
                "use_lam": True,
                "lam_ckpt": str(lam),
                "lam_model_dim": 32,
                "lam_latent_dim": 8,
                "lam_num_latents": 6,
            }
        },
    )
    result = apply_ckpt_config_to_demo(
        DemoArgs("rollout", "unused", "unused", rollout=RolloutArgs(dyn_ckpt=str(dyn)))
    )
    assert result.rollout.lam.use_lam and result.rollout.lam.lam_ckpt == str(lam)
    assert (
        result.rollout.lam.lam_model_dim == 32
        and result.rollout.lam.lam_num_latents == 6
    )


def test_pipeline_b_checks_split_before_loading_models(tmp_path, monkeypatch):
    from configs.eval_configs import RolloutConfig
    from fluidgenie.eval.rollout_runner import run_rollout
    import fluidgenie.data.splits as splits

    dyn = tmp_path / "dyn"
    save_config_json(dyn, {"config": {"architecture_version": 2, "data": "unused"}})
    called = []

    def reject(config, directory):
        called.append(config)
        raise ValueError("overlap regression sentinel")

    monkeypatch.setattr(splits, "assert_checkpoint_evaluation", reject)
    with pytest.raises(ValueError, match="overlap regression sentinel"):
        run_rollout(
            RolloutConfig(
                str(tmp_path / "episode.npy"), "unused", str(dyn), str(tmp_path / "out")
            )
        )
    assert len(called) == 1


@pytest.mark.parametrize(
    "path,body,length,expected",
    [
        ("/reset", b"", 0, 200),
        ("/step", b"null", 4, 400),
        ("/step", b"[]", 2, 400),
        ("/step", b'{"action":null}', 15, 400),
        ("/unknown", b"{}", 2, 404),
        ("/step", b"", -1, 400),
    ],
)
def test_demo_http_errors_and_empty_reset_without_network(path, body, length, expected):
    from fluidgenie.cli.steering_demo import make_handler

    class Session:
        steps = 0

        def reset(self):
            return np.zeros((4, 4, 3))

        def step(self, action):
            return np.zeros((4, 4, 3))

    Handler = make_handler(Session(), b"page")
    handler = Handler.__new__(Handler)
    handler.path = path
    handler.headers = {"Content-Length": str(length)}
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    statuses = []
    handler.send_response = lambda code: statuses.append(code)
    handler.send_error = lambda code: statuses.append(code)
    handler.send_header = lambda *_: None
    handler.end_headers = lambda: None
    handler.do_POST()
    assert statuses == [expected]


def test_lam_architecture_and_length_guards_before_loading(tmp_path):
    from fluidgenie.models.lam_conditioning import load_frozen_lam

    save_config_json(
        tmp_path, {"kind": "lam", "config": {"architecture_version": 1, "seq_len": 4}}
    )
    with pytest.raises(ValueError, match="architecture_version"):
        load_frozen_lam(str(tmp_path), jnp.zeros((1, 3, 4, 4, 3)))
    save_config_json(
        tmp_path, {"kind": "lam", "config": {"architecture_version": 2, "seq_len": 4}}
    )
    with pytest.raises(ValueError, match="trained seq_len"):
        load_frozen_lam(str(tmp_path), jnp.zeros((1, 5, 4, 4, 3)))


def test_microbatch_logs_bounded_residual_saturation_and_gradients():
    from fluidgenie.models.continuous_residual import ContinuousResidualUNet
    from fluidgenie.training.train_continuous_residual import make_microbatch_train_step
    from fluidgenie.training.optimizers import make_optimizer

    model = ContinuousResidualUNet(3, 2, (4, 8, 8), 1, (0.1, 0.1, 0.1), True)
    context = jnp.ones((2, 2, 4, 4, 3), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), context)["params"]
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=make_optimizer(1e-3, 20)
    )
    _, metrics = make_microbatch_train_step(1, capture_saturation=True)(
        state, context, jnp.zeros((2, 2, 4, 4, 3)), jax.random.PRNGKey(1)
    )
    assert (
        "tanh_sat_frac" in metrics
        and np.isfinite(metrics["grad_norm"])
        and metrics["grad_norm"] > 0
    )


def test_matched_benchmark_loads_once_and_reuses_maximum_horizon(tmp_path, monkeypatch):
    import scripts.benchmark_repaired_models as benchmark

    train, test = tmp_path / "train", tmp_path / "test"
    for seed, directory in [(1, train), (2, test)]:
        directory.mkdir()
        np.save(
            directory / "episode_000000.npy",
            np.full((8, 4, 4, 3), seed, dtype=np.float32),
        )
        (directory / "episode_000000.json").write_text(
            json.dumps({"seed": seed, "dt": 0.02, "substeps": 5})
        )
    counts = {"cr": 0, "fm": 0}

    class CR:
        def apply(self, variables, context, training=False):
            return jnp.zeros_like(context[:, -1])

    class FM:
        action_dim = 0

        def apply(self, variables, x, tau, context, **kwargs):
            return jnp.zeros_like(x)

    def cr_loader(*args):
        counts["cr"] += 1
        return CR(), {}, {"config": {"context": 2}}

    def fm_loader(*args):
        counts["fm"] += 1
        return FM(), {}

    monkeypatch.setattr(benchmark, "_load_residual_model", cr_loader)
    monkeypatch.setattr(benchmark, "_load_model", fm_loader)
    monkeypatch.setattr(
        benchmark, "load_config_json", lambda _: {"config": {"context": 2}}
    )
    stats = tmp_path / "stats.npz"
    np.savez(stats, mean=np.zeros(3), std=np.ones(3))
    out = tmp_path / "out"
    result = benchmark.main(
        benchmark.Args(
            str(train),
            str(test),
            str(stats),
            "cr",
            "fm",
            str(out),
            horizons=(2, 5),
            episodes=1,
            ode_steps=2,
        )
    )
    assert counts == {"cr": 1, "fm": 1} and len(result["episodes"]) == 2
    for model in ["cr", "fm"]:
        small = np.load(out / model / "episode_000000/h2/rollout_arrays.npz")["pred"]
        large = np.load(out / model / "episode_000000/h5/rollout_arrays.npz")["pred"]
        assert np.array_equal(small, large[:2])
    json.loads(
        (out / "benchmark_summary.json").read_text(),
        parse_constant=lambda x: pytest.fail("Nonstandard JSON constant " + x),
    )


def test_residual_evaluation_reuses_runtime_and_emits_physics_summary(
    tmp_path, monkeypatch
):
    import fluidgenie.eval.continuous_residual_rollout as evaluation

    data = tmp_path / "data"
    data.mkdir()
    fields = np.ones((6, 4, 4, 3), dtype=np.float32)
    np.save(data / "episode_000000.npy", fields)
    saved = {"config": {"context": 2}}
    calls = []
    monkeypatch.setattr(
        evaluation,
        "_load_residual_model",
        lambda *args: pytest.fail("Runtime should be reused"),
    )
    runtime = (saved, lambda context: context[:, -1], None)
    result = evaluation.run_residual_rollout(
        str(data / "episode_000000.npy"),
        "unused",
        str(tmp_path / "out"),
        "",
        0,
        2,
        2,
        False,
        runtime=runtime,
        split_validation={"disjoint": True},
    )
    assert (
        "physics_summary" in result
        and result["physics_summary"]["vorticity_correlation"] == 1
    )
    cfg = evaluation.ContinuousResidualEvaluationConfig(
        str(data),
        "unused",
        str(tmp_path / "teacher"),
        episode_indices=(0,),
        teacher_target_indices=(2, 3),
        make_visualizations=False,
    )
    teacher = evaluation.run_teacher_forced(
        cfg, runtime=runtime, split_validation={"disjoint": True}
    )
    assert teacher["aggregate"]["mse"] == 0


def test_pipeline_b_real_tiny_checkpoints_complete_synthetic_rollout(
    tmp_path, monkeypatch
):
    from fluidgenie.models.tokenizer_st import TokenizerSTVQVAE
    from fluidgenie.models.dynamics_st import DynamicsSTMaskGIT
    from fluidgenie.models.lam import LatentActionModel
    from fluidgenie.training.checkpoint_utils import save_params
    from configs.eval_configs import RolloutConfig
    import fluidgenie.eval.rollout_runner as runner

    video = jnp.zeros((1, 4, 4, 4, 3), dtype=jnp.float32)
    tok = TokenizerSTVQVAE(3, 8, 4, 8, 2, 1, 2, 0.0, 0.0, 0.0)
    tok_params = tok.init(jax.random.PRNGKey(0), {"videos": video}, training=False)[
        "params"
    ]
    lam = LatentActionModel(3, 8, 4, 8, 2, 1, 2, 0.0, 0.0, architecture_version=3)
    lam_params = lam.init(jax.random.PRNGKey(1), {"videos": video}, training=False)[
        "params"
    ]
    dyn = DynamicsSTMaskGIT(
        model_dim=8,
        num_latents=8,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
        mask_ratio_min=0.0,
        mask_ratio_max=1.0,
        mask_full_prob=0.2,
    )
    batch = {
        "video_tokens": jnp.zeros((1, 3, 2, 2), dtype=jnp.int32),
        "mask_rng": jax.random.PRNGKey(2),
        "latent_actions": jnp.zeros((1, 2, 1, 4)),
    }
    dyn_params = dyn.init(jax.random.PRNGKey(3), batch, training=False)["params"]
    for name, params, cfg in [
        ("tok", tok_params, {"architecture_version": 2}),
        ("lam", lam_params, {"architecture_version": 3, "seq_len": 4}),
        (
            "dyn",
            dyn_params,
            {"architecture_version": 2, "context": 2, "model": "st_maskgit"},
        ),
    ]:
        root = tmp_path / name
        save_config_json(root, {"config": cfg})
        save_params(root, "latest", params)
    episode = tmp_path / "episode.npy"
    np.save(episode, np.zeros((4, 4, 4, 3), dtype=np.float32))
    monkeypatch.setattr(runner, "visualize_rollout", lambda *a, **kw: None)
    cfg = RolloutConfig(
        str(episode),
        str(tmp_path / "tok/latest"),
        str(tmp_path / "dyn/latest"),
        str(tmp_path / "out"),
        horizon=2,
        context=2,
        tokenizer_arch="st",
        model_type="st_maskgit",
        codebook_size=8,
        embed_dim=4,
        patch_size=2,
        model_dim=8,
        num_blocks=1,
        num_heads_tok=2,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        mask_steps=2,
        use_lam=True,
        lam_ckpt=str(tmp_path / "lam/latest"),
        lam_model_dim=8,
        lam_latent_dim=4,
        lam_num_latents=8,
        lam_patch_size=2,
        lam_num_blocks=1,
        lam_num_heads=2,
        skip_split_check=True,
    )
    runner.run_rollout(cfg)
    assert np.isfinite(np.load(tmp_path / "out/rollout_arrays.npz")["pred"]).all()
    split = json.loads((tmp_path / "out/split_validation.json").read_text())
    assert not split["disjoint"] and "synthetic" in split["status"]


def test_discovery_certificate_rejects_old_intra_window_protocol():
    from fluidgenie.eval.action_discovery import require_discovery_certificate

    certificate = dict(
        discovery_gate_passed=True, five_action_mapping_gate=True, gate={"version": 3}
    )
    with pytest.raises(ValueError, match="episode-wide"):
        require_discovery_certificate(certificate)
    certificate.update(
        gate={"version": 4}, shuffle_protocol="episode_wide_transition_permutation_v4"
    )
    require_discovery_certificate(certificate)
