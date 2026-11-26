"""Evaluation-only sampling controls for the frozen FM0 checkpoint.

This module never trains or changes FM0.  It measures whether averaging
independent conditional-flow samples, or reusing one source-noise key across
an autoregressive rollout, explains the observed overactivity.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from configs.config_io import load_config_json

from fluidgenie.eval.flow_matching_diagnostics import _load_model, _mean_abs
from fluidgenie.eval.flow_matching_rollout import _denorm, _load_stats, _norm
from fluidgenie.eval.metrics import compute_rollout_metrics
from fluidgenie.eval.rollout_runner import visualize_rollout
from fluidgenie.eval.samplers.flow_matching import sample_flow_matching_heun


@dataclass
class FM0SamplingControlConfig:
    dyn_ckpt: str
    data_dir: str
    stats_path: str
    out_dir: str
    episode_indices: Tuple[int, int, int] = (0, 1, 2)
    teacher_target_indices: Tuple[int, ...] = (2, 12, 22, 32, 42, 52)
    sample_counts: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    rollout_sample_counts: Tuple[int, ...] = (1, 4, 8, 16)
    horizon: int = 40
    ode_steps: int = 8
    held_action_id: int | None = None
    base_seed: int = 1000


def _robust_motion_rows(rows: list[dict]) -> dict:
    gt = np.asarray([row["gt_motion"] for row in rows], dtype=np.float64)
    pred = np.asarray([row["pred_motion"] for row in rows], dtype=np.float64)
    ratio = pred / np.maximum(gt, 1e-8)
    return {
        "gt_motion_mean": float(gt.mean()),
        "pred_motion_mean": float(pred.mean()),
        "motion_ratio_of_means": float(pred.mean() / max(gt.mean(), 1e-8)),
        "motion_ratio_median": float(np.median(ratio)),
        "motion_ratio_mean": float(ratio.mean()),
    }


def _sample_many(sample_fn, context: np.ndarray, keys: jax.Array, stats) -> np.ndarray:
    samples = [
        _denorm(
            np.asarray(sample_fn(jnp.asarray(context[None], dtype=jnp.float32), key))[
                0
            ][None],
            stats,
        )[0]
        for key in keys
    ]
    return np.asarray(samples, dtype=np.float32)


def _write_rollout(out: Path, gt: np.ndarray, pred: np.ndarray, metadata: dict) -> dict:
    metrics = compute_rollout_metrics(gt, pred, view="density")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "rollout_arrays.npz", gt=gt, pred=pred)
    payload = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "frame",
            "mse",
            "mae",
            "density_mae",
            "gt_motion",
            "pred_motion",
            "motion_ratio",
            "is_frozen",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in range(len(metrics["mse"])):
            writer.writerow(
                {
                    "frame": frame,
                    "mse": float(metrics["mse"][frame]),
                    "mae": float(metrics["mae"][frame]),
                    "density_mae": float(metrics["view_mae"][frame]),
                    "gt_motion": float(metrics["gt_motion"][frame]),
                    "pred_motion": float(metrics["pred_motion"][frame]),
                    "motion_ratio": float(metrics["motion_ratio"][frame]),
                    "is_frozen": bool(metrics["is_frozen"][frame]),
                }
            )
    (out / "rollout_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    for view in ("density", "speed", "vorticity"):
        view_out = out / view
        view_out.mkdir(parents=True, exist_ok=True)
        visualize_rollout(gt, pred, view_out, view=view)
    return {
        k: float(metrics[k])
        for k in (
            "mse_mean",
            "mae_mean",
            "view_mae_mean",
            "gt_motion_mean",
            "pred_motion_mean",
            "motion_ratio_of_means",
            "motion_ratio_median",
            "motion_ratio_mean",
            "frozen_fraction",
        )
    }


def run_controls(cfg: FM0SamplingControlConfig) -> dict:
    out, data_dir = Path(cfg.out_dir), Path(cfg.data_dir)
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
        lambda ctx, key: sample_flow_matching_heun(
            model, params, ctx, key, cfg.ode_steps, action_cond=action_cond
        )
    )

    one_step: dict[str, dict] = {}
    for count in cfg.sample_counts:
        rows = []
        for episode in cfg.episode_indices:
            fields = np.asarray(
                np.load(data_dir / f"episode_{episode:06d}.npy", mmap_mode="r"),
                dtype=np.float32,
            )
            for target_index in cfg.teacher_target_indices:
                context_raw = fields[target_index - 2 : target_index]
                target = fields[target_index]
                keys = jax.random.split(
                    jax.random.PRNGKey(
                        cfg.base_seed + episode * 100_000 + target_index * 1000
                    ),
                    count,
                )
                samples = _sample_many(
                    sample_fn, _norm(context_raw, stats), keys, stats
                )
                prediction = samples.mean(axis=0)
                rows.append(
                    {
                        "episode_index": episode,
                        "target_index": target_index,
                        "mse": float(np.mean((prediction - target) ** 2)),
                        "mae": _mean_abs(prediction - target),
                        "density_mae": _mean_abs(prediction[..., 2] - target[..., 2]),
                        "prediction_variance": float(samples.var(axis=0).mean()),
                        "gt_motion": _mean_abs(target - context_raw[-1]),
                        "pred_motion": _mean_abs(prediction - context_raw[-1]),
                    }
                )
        aggregate = {
            name: float(np.mean([row[name] for row in rows]))
            for name in (
                "mse",
                "mae",
                "density_mae",
                "prediction_variance",
            )
        }
        aggregate.update(_robust_motion_rows(rows))
        one_step[str(count)] = {"aggregate": aggregate, "rows": rows}

    rollouts: dict[str, dict] = {}
    for count in cfg.rollout_sample_counts:
        episodes = []
        for episode in cfg.episode_indices:
            fields = np.asarray(
                np.load(data_dir / f"episode_{episode:06d}.npy", mmap_mode="r"),
                dtype=np.float32,
            )
            context = jnp.asarray(_norm(fields[:2], stats)[None], dtype=jnp.float32)
            rng = jax.random.PRNGKey(cfg.base_seed + episode + 1)
            prediction_frames = []
            for _ in range(cfg.horizon):
                keys = jax.random.split(rng, count + 1)
                rng, sample_keys = keys[0], keys[1:]
                samples = [sample_fn(context, key) for key in sample_keys]
                next_norm = jnp.mean(jnp.stack(samples, axis=0), axis=0)
                prediction_frames.append(
                    _denorm(np.asarray(next_norm[0])[None], stats)[0]
                )
                context = jnp.concatenate([context[:, 1:], next_norm[:, None]], axis=1)
            pred = np.asarray(prediction_frames, dtype=np.float32)
            gt = fields[2 : 2 + cfg.horizon]
            episode_out = (
                out / "rollout_mean" / f"k_{count:02d}" / f"episode_{episode:06d}"
            )
            metrics = _write_rollout(
                episode_out,
                gt,
                pred,
                {
                    "mode": "conditional_mean",
                    "sample_count": count,
                    "episode_index": episode,
                    "seed": cfg.base_seed + episode,
                    "horizon": cfg.horizon,
                    "ode_steps": cfg.ode_steps,
                },
            )
            episodes.append(
                {"episode_index": episode, "seed": cfg.base_seed + episode, **metrics}
            )
        rollouts[str(count)] = {
            "episodes": episodes,
            "aggregate": {
                name: {
                    "mean": float(np.mean([x[name] for x in episodes])),
                    "std": float(np.std([x[name] for x in episodes])),
                }
                for name in episodes[0]
                if name not in {"episode_index", "seed"}
            },
        }

    # E1: exactly one source-noise key is reused at every autoregressive step.
    fixed_noise = []
    for episode in cfg.episode_indices:
        fields = np.asarray(
            np.load(data_dir / f"episode_{episode:06d}.npy", mmap_mode="r"),
            dtype=np.float32,
        )
        context = jnp.asarray(_norm(fields[:2], stats)[None], dtype=jnp.float32)
        key = jax.random.PRNGKey(cfg.base_seed + 50_000 + episode)
        prediction_frames = []
        for _ in range(cfg.horizon):
            next_norm = sample_fn(context, key)
            prediction_frames.append(_denorm(np.asarray(next_norm[0])[None], stats)[0])
            context = jnp.concatenate([context[:, 1:], next_norm[:, None]], axis=1)
        pred, gt = (
            np.asarray(prediction_frames, dtype=np.float32),
            fields[2 : 2 + cfg.horizon],
        )
        metrics = _write_rollout(
            out / "fixed_noise" / f"episode_{episode:06d}",
            gt,
            pred,
            {
                "mode": "fixed_source_noise",
                "episode_index": episode,
                "seed": cfg.base_seed + 50_000 + episode,
                "horizon": cfg.horizon,
                "ode_steps": cfg.ode_steps,
            },
        )
        fixed_noise.append(
            {
                "episode_index": episode,
                "seed": cfg.base_seed + 50_000 + episode,
                **metrics,
            }
        )

    result = {
        "config": asdict(cfg),
        "one_step_conditional_mean": one_step,
        "recurrent_conditional_mean": rollouts,
        "fixed_noise_rollout": {
            "episodes": fixed_noise,
            "aggregate": {
                name: {
                    "mean": float(np.mean([x[name] for x in fixed_noise])),
                    "std": float(np.std([x[name] for x in fixed_noise])),
                }
                for name in fixed_noise[0]
                if name not in {"episode_index", "seed"}
            },
        },
    }
    (out / "sampling_controls.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print(f"Saved FM0 sampling controls to {out}")
    return result


def main() -> None:
    run_controls(tyro.cli(FM0SamplingControlConfig))


if __name__ == "__main__":
    main()
