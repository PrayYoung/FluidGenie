"""Post-training FM0 diagnostics: teacher forcing, sampling variation, and drift."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from configs.config_io import load_config_json
from fluidgenie.eval.flow_matching_rollout import _denorm, _load_stats, _norm
from fluidgenie.eval.samplers.flow_matching import sample_flow_matching_heun
from fluidgenie.models.flow_matching import ConditionalFlowUNet
from fluidgenie.training.checkpoint_utils import load_params


@dataclass
class FM0DiagnosticConfig:
    dyn_ckpt: str
    data_dir: str
    stats_path: str
    formal_eval_dir: str
    out_dir: str
    episode_indices: Tuple[int, int, int] = (0, 1, 2)
    # Target frame indices.  A transition i uses fields[i-2:i] -> fields[i].
    teacher_target_indices: Tuple[int, ...] = (2, 12, 22, 32, 42, 52)
    multiseed_target_indices: Tuple[int, ...] = (12, 52)
    num_seeds: int = 16
    ode_steps: int = 8
    held_action_id: int | None = None
    base_seed: int = 2000


def _mean_abs(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def _load_model(ckpt: str, h: int, w: int, channels: int):
    saved = load_config_json(ckpt)
    if not saved or saved.get("kind") != "flow_matching_fm0":
        raise ValueError("dyn_ckpt must be an FM0 checkpoint")
    cfg = saved["config"]
    if int(cfg["context"]) != 2:
        raise ValueError("FM0 diagnostics require context=2")
    model = ConditionalFlowUNet(
        channels=channels,
        context_frames=2,
        hidden_dims=tuple(cfg["hidden_dims"]),
        time_dim=int(cfg["time_dim"]),
        num_res_blocks=int(cfg["num_res_blocks"]),
        modern=bool(cfg.get("modern", False)),
        action_dim=int(cfg.get("action_dim", 0)),
    )
    init = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, h, w, channels), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.zeros((1, 2, h, w, channels), dtype=jnp.float32),
        training=False,
        **(
            {"action_cond": jnp.zeros((1, model.action_dim), dtype=jnp.float32)}
            if model.action_dim
            else {}
        ),
    )["params"]
    return model, load_params(ckpt, init)


def _sample(sample_fn, context: np.ndarray, key: jax.Array, stats) -> np.ndarray:
    next_norm = np.asarray(
        sample_fn(jnp.asarray(context[None], dtype=jnp.float32), key)
    )[0]
    return _denorm(next_norm[None], stats)[0].astype(np.float32)


def _horizon_table(data_dir: Path, formal_eval_dir: Path, episodes: Tuple[int, ...]):
    requested = (1, 2, 3, 5, 10, 20, 40)
    per_episode = []
    for episode in episodes:
        arrays = np.load(
            formal_eval_dir / f"episode_{episode:06d}" / "rollout_arrays.npz"
        )
        gt, pred = arrays["gt"], arrays["pred"]
        fields = np.load(data_dir / f"episode_{episode:06d}.npy", mmap_mode="r")
        previous = np.asarray(fields[1], dtype=np.float32)  # formal start=0, context=2
        rows = []
        for horizon in requested:
            i = horizon - 1
            gt_prev = previous if i == 0 else gt[i - 1]
            pred_prev = previous if i == 0 else pred[i - 1]
            gt_motion = _mean_abs(gt[i] - gt_prev)
            pred_motion = _mean_abs(pred[i] - pred_prev)
            rows.append(
                {
                    "horizon": horizon,
                    "mse": float(np.mean((pred[i] - gt[i]) ** 2)),
                    "gt_motion": gt_motion,
                    "pred_motion": pred_motion,
                    "motion_ratio": pred_motion / max(gt_motion, 1e-8),
                }
            )
        per_episode.append({"episode_index": episode, "rows": rows})
    aggregate = []
    for i, horizon in enumerate(requested):
        values = [entry["rows"][i] for entry in per_episode]
        aggregate.append(
            {
                "horizon": horizon,
                **{
                    name: float(np.mean([row[name] for row in values]))
                    for name in ("mse", "gt_motion", "pred_motion", "motion_ratio")
                },
            }
        )
    return {"per_episode": per_episode, "aggregate": aggregate}


def run_diagnostics(cfg: FM0DiagnosticConfig) -> dict:
    data_dir, out = Path(cfg.data_dir), Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    first = np.load(
        data_dir / f"episode_{cfg.episode_indices[0]:06d}.npy", mmap_mode="r"
    )
    _, h, w, channels = first.shape
    model, params = _load_model(cfg.dyn_ckpt, h, w, channels)
    if model.action_dim:
        saved = load_config_json(cfg.dyn_ckpt)
        if (
            saved["config"].get("use_lam")
            or model.action_dim != 5
            or cfg.held_action_id not in range(5)
        ):
            raise ValueError(
                "Diagnostics require an explicit --held-action-id for supervised checkpoints; use the LAM rollout evaluator for latent checkpoints"
            )
        action_cond = jax.nn.one_hot(
            jnp.asarray([cfg.held_action_id]), 5, dtype=jnp.float32
        )
    else:
        action_cond = None
    from fluidgenie.eval.utils import resolve_stats_path
    from fluidgenie.data.splits import assert_checkpoint_evaluation

    assert_checkpoint_evaluation(load_config_json(cfg.dyn_ckpt)["config"], data_dir)
    stats = _load_stats(resolve_stats_path(cfg.stats_path, cfg.dyn_ckpt))
    sample_fn = jax.jit(
        lambda context, key: sample_flow_matching_heun(
            model, params, context, key, cfg.ode_steps, action_cond=action_cond
        )
    )

    teacher_rows = []
    multi_rows = []
    key_counter = 0
    for episode in cfg.episode_indices:
        fields = np.asarray(
            np.load(data_dir / f"episode_{episode:06d}.npy", mmap_mode="r"),
            dtype=np.float32,
        )
        for target_index in cfg.teacher_target_indices:
            context_raw, target = (
                fields[target_index - 2 : target_index],
                fields[target_index],
            )
            context = _norm(context_raw, stats)
            prediction = _sample(
                sample_fn,
                context,
                jax.random.PRNGKey(cfg.base_seed + key_counter),
                stats,
            )
            key_counter += 1
            gt_motion = _mean_abs(target - context_raw[-1])
            pred_motion = _mean_abs(prediction - context_raw[-1])
            teacher_rows.append(
                {
                    "episode_index": episode,
                    "target_index": target_index,
                    "mse": float(np.mean((prediction - target) ** 2)),
                    "mae": _mean_abs(prediction - target),
                    "density_mae": _mean_abs(prediction[..., 2] - target[..., 2]),
                    "gt_motion": gt_motion,
                    "pred_motion": pred_motion,
                    "motion_ratio": pred_motion / max(gt_motion, 1e-8),
                }
            )
        for target_index in cfg.multiseed_target_indices:
            context_raw, target = (
                fields[target_index - 2 : target_index],
                fields[target_index],
            )
            context = _norm(context_raw, stats)
            samples = np.stack(
                [
                    _sample(
                        sample_fn,
                        context,
                        jax.random.PRNGKey(cfg.base_seed + 100000 + key_counter + seed),
                        stats,
                    )
                    for seed in range(cfg.num_seeds)
                ]
            )
            key_counter += cfg.num_seeds
            sample_mean = samples.mean(axis=0)
            sample_std = float(samples.std(axis=0).mean())
            differences = samples[:, None] - samples[None, :]
            pairwise = np.mean(np.abs(differences), axis=(2, 3, 4))
            pairwise_mean = float(pairwise[np.triu_indices(cfg.num_seeds, 1)].mean())
            true_motion = _mean_abs(target - context_raw[-1])
            multi_rows.append(
                {
                    "episode_index": episode,
                    "target_index": target_index,
                    "num_seeds": cfg.num_seeds,
                    "per_element_sample_std": sample_std,
                    "pairwise_prediction_distance": pairwise_mean,
                    "true_motion": true_motion,
                    "std_to_true_motion": sample_std / max(true_motion, 1e-8),
                    "pairwise_to_true_motion": pairwise_mean / max(true_motion, 1e-8),
                    "individual_sample_mse": float(
                        np.mean((samples - target[None]) ** 2)
                    ),
                    "sample_mean_mse": float(np.mean((sample_mean - target) ** 2)),
                }
            )

    def mean_rows(rows, names):
        return {name: float(np.mean([row[name] for row in rows])) for name in names}

    teacher = {
        "rows": teacher_rows,
        "aggregate": mean_rows(
            teacher_rows,
            ("mse", "mae", "density_mae", "gt_motion", "pred_motion", "motion_ratio"),
        ),
    }
    multiseed = {
        "rows": multi_rows,
        "aggregate": mean_rows(
            multi_rows,
            (
                "per_element_sample_std",
                "pairwise_prediction_distance",
                "true_motion",
                "std_to_true_motion",
                "pairwise_to_true_motion",
                "individual_sample_mse",
                "sample_mean_mse",
            ),
        ),
    }
    result = {
        "config": asdict(cfg),
        "teacher_forced": teacher,
        "multiseed": multiseed,
        "horizon_wise": _horizon_table(
            data_dir, Path(cfg.formal_eval_dir), cfg.episode_indices
        ),
    }
    (out / "diagnostics.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved FM0 diagnostics to {out}")
    return result


def main() -> None:
    run_diagnostics(tyro.cli(FM0DiagnosticConfig))


if __name__ == "__main__":
    main()
