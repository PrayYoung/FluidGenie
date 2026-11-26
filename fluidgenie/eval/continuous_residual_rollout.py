"""Evaluation and frozen multi-episode protocol for CR0/CR1 residual dynamics."""

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
from fluidgenie.eval.flow_matching_rollout import _denorm, _load_stats, _norm
from fluidgenie.eval.metrics import (
    compute_rollout_metrics,
    fluid_physics_metrics,
    jsonable,
    episode_physics_metadata,
    summarize_physics,
)
from fluidgenie.data.splits import assert_checkpoint_evaluation
from fluidgenie.eval.rollout_runner import visualize_rollout
from fluidgenie.models.continuous_residual import ContinuousResidualUNet
from fluidgenie.training.checkpoint_utils import load_params


@dataclass
class ContinuousResidualEvaluationConfig:
    data_dir: str
    dyn_ckpt: str
    out_dir: str
    episode_indices: Tuple[int, ...] = (0, 1, 2)
    start: int = 0
    horizon: int = 40
    context: int = 2
    stats_path: str = ""
    teacher_target_indices: Tuple[int, ...] = (2, 12, 22, 32, 42, 52)
    make_visualizations: bool = True


_SCALAR = (
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


def _load_residual_model(ckpt: str, h: int, w: int, channels: int):
    saved = load_config_json(ckpt)
    if not saved or saved.get("kind") not in {
        "continuous_residual_cr0",
        "continuous_residual_cr1_k4",
        "continuous_residual_cr1_v2_k4_bounded",
        "continuous_residual_cr1_v3_noise",
        "continuous_residual_cr1_v3_k8",
        "continuous_residual_cr1_physics",
    }:
        raise ValueError("dyn_ckpt must be a CR0 or CR1 checkpoint")
    cfg = saved["config"]
    model = ContinuousResidualUNet(
        channels=channels,
        context_frames=int(cfg["context"]),
        hidden_dims=tuple(cfg["hidden_dims"]),
        num_res_blocks=int(cfg["num_res_blocks"]),
        residual_scales=tuple(cfg.get("residual_scales", ())),
        stable=bool(cfg.get("stable", False)),
    )
    init = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, int(cfg["context"]), h, w, channels), dtype=jnp.float32),
        training=False,
    )["params"]
    return model, load_params(ckpt, init), saved


def load_residual_runtime(ckpt, h, w, channels, stats_path):
    from fluidgenie.eval.utils import resolve_stats_path

    model, params, saved = _load_residual_model(ckpt, h, w, channels)
    step = jax.jit(
        lambda ctx: ctx[:, -1] + model.apply({"params": params}, ctx, training=False)
    )
    stats = _load_stats(resolve_stats_path(stats_path, ckpt))
    return saved, step, stats


def run_residual_rollout(
    npy_path: str,
    dyn_ckpt: str,
    out_dir: str,
    stats_path: str,
    start: int,
    horizon: int,
    context_frames: int,
    make_visualizations: bool,
    runtime=None,
    split_validation=None,
) -> dict:
    fields = np.load(npy_path, mmap_mode="r")
    total, h, w, channels = fields.shape
    if (
        start < 0
        or context_frames < 1
        or horizon < 1
        or start + context_frames + horizon > total
    ):
        raise ValueError("Episode too short for frozen residual rollout")
    saved, step, stats = runtime or load_residual_runtime(
        dyn_ckpt, h, w, channels, stats_path
    )
    if context_frames != int(saved["config"]["context"]):
        raise ValueError("Rollout context does not match checkpoint")
    split = split_validation or assert_checkpoint_evaluation(
        saved["config"], Path(npy_path).parent
    )
    context = jnp.asarray(
        _norm(
            np.asarray(fields[start : start + context_frames], dtype=np.float32), stats
        )[None],
        dtype=jnp.float32,
    )
    prediction_frames = []
    for _ in range(horizon):
        next_norm = step(context)
        prediction_frames.append(_denorm(np.asarray(next_norm[0])[None], stats)[0])
        context = jnp.concatenate([context[:, 1:], next_norm[:, None]], axis=1)
    gt, pred = (
        np.asarray(fields[start + context_frames : start + context_frames + horizon]),
        np.asarray(prediction_frames, dtype=np.float32),
    )
    metrics = compute_rollout_metrics(gt, pred, view="density")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "rollout_arrays.npz", gt=gt, pred=pred)
    physics = fluid_physics_metrics(
        gt,
        pred,
        last_context=fields[start + context_frames - 1],
        **episode_physics_metadata(npy_path),
    )
    metrics["physics_summary"] = summarize_physics(physics)
    (out / "physics_metrics.json").write_text(
        json.dumps(jsonable(physics), indent=2, allow_nan=False)
    )
    (out / "split_validation.json").write_text(json.dumps(split, indent=2))
    payload = jsonable(metrics)
    (out / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    )
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "mse",
                "mae",
                "density_mae",
                "gt_motion",
                "pred_motion",
                "motion_ratio",
                "is_frozen",
            ],
        )
        writer.writeheader()
        for frame in range(horizon):
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
    if make_visualizations:
        for view in ("density", "speed", "vorticity"):
            view_out = out / view
            view_out.mkdir(parents=True, exist_ok=True)
            visualize_rollout(gt, pred, view_out, view=view)
    return metrics


def run_teacher_forced(
    cfg: ContinuousResidualEvaluationConfig, runtime=None, split_validation=None
) -> dict:
    first = np.load(
        Path(cfg.data_dir) / f"episode_{cfg.episode_indices[0]:06d}.npy", mmap_mode="r"
    )
    _, h, w, channels = first.shape
    saved, step, stats = runtime or load_residual_runtime(
        cfg.dyn_ckpt, h, w, channels, cfg.stats_path
    )
    if cfg.context != int(saved["config"]["context"]) or channels < 3:
        raise ValueError(
            "Teacher forcing requires matching context and at least three channels"
        )
    if split_validation is None:
        assert_checkpoint_evaluation(saved["config"], cfg.data_dir)
    rows = []
    for episode in cfg.episode_indices:
        fields = np.asarray(
            np.load(Path(cfg.data_dir) / f"episode_{episode:06d}.npy", mmap_mode="r"),
            dtype=np.float32,
        )
        for target_index in cfg.teacher_target_indices:
            if not cfg.context <= target_index < len(fields):
                raise ValueError(
                    "Teacher-forcing target index is outside episode bounds"
                )
            raw_context, target = (
                fields[target_index - cfg.context : target_index],
                fields[target_index],
            )
            if raw_context.shape[0] != cfg.context:
                raise ValueError(
                    f"Target index {target_index} lacks {cfg.context} context frames"
                )
            prediction = _denorm(
                np.asarray(
                    step(
                        jnp.asarray(_norm(raw_context, stats)[None], dtype=jnp.float32)
                    )[0][None]
                ),
                stats,
            )[0]
            rows.append(
                {
                    "episode_index": episode,
                    "target_index": target_index,
                    "mse": float(np.mean((prediction - target) ** 2)),
                    "mae": float(
                        np.mean(np.abs(prediction - target)),
                    ),
                    "density_mae": float(
                        np.mean(np.abs(prediction[..., 2] - target[..., 2])),
                    ),
                    "gt_motion": float(np.mean(np.abs(target - raw_context[-1]))),
                    "pred_motion": float(np.mean(np.abs(prediction - raw_context[-1]))),
                }
            )
    gt, pred = (
        np.asarray([row["gt_motion"] for row in rows]),
        np.asarray([row["pred_motion"] for row in rows]),
    )
    ratios = pred / np.maximum(gt, 1e-8)
    return {
        "rows": rows,
        "aggregate": {
            **{
                name: float(np.mean([row[name] for row in rows]))
                for name in ("mse", "mae", "density_mae")
            },
            "gt_motion_mean": float(gt.mean()),
            "pred_motion_mean": float(pred.mean()),
            "motion_ratio_of_means": float(pred.mean() / max(gt.mean(), 1e-8)),
            "motion_ratio_median": float(np.median(ratios)),
            "motion_ratio_mean": float(ratios.mean()),
        },
    }


def main() -> None:
    cfg = tyro.cli(ContinuousResidualEvaluationConfig)
    if cfg.context not in (1, 2) or cfg.horizon < 1:
        raise ValueError(
            "Continuous residual evaluation requires context=1 or 2 and a positive horizon"
        )
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    first = np.load(
        Path(cfg.data_dir) / f"episode_{cfg.episode_indices[0]:06d}.npy", mmap_mode="r"
    )
    runtime = load_residual_runtime(cfg.dyn_ckpt, *first.shape[1:], cfg.stats_path)
    split_validation = assert_checkpoint_evaluation(runtime[0]["config"], cfg.data_dir)
    rows = []
    for episode in cfg.episode_indices:
        metrics = run_residual_rollout(
            str(Path(cfg.data_dir) / f"episode_{episode:06d}.npy"),
            cfg.dyn_ckpt,
            str(out / f"episode_{episode:06d}"),
            cfg.stats_path,
            cfg.start,
            cfg.horizon,
            cfg.context,
            cfg.make_visualizations,
            runtime=runtime,
            split_validation=split_validation,
        )
        rows.append(
            {
                "episode_index": episode,
                **{key: float(metrics[key]) for key in _SCALAR},
                **{"physics_" + k: v for k, v in metrics["physics_summary"].items()},
            }
        )
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in _SCALAR
    }
    physics_keys = [key for key in rows[0] if key.startswith("physics_")]
    for key in physics_keys:
        valid = [
            row[key] for row in rows if row[key] is not None and np.isfinite(row[key])
        ]
        aggregate[key] = {
            "mean": float(np.mean(valid)) if valid else None,
            "std": float(np.std(valid)) if valid else None,
            "count": len(valid),
            "missing_count": len(rows) - len(valid),
        }
    teacher_forced = run_teacher_forced(
        cfg, runtime=runtime, split_validation=split_validation
    )
    (out / "evaluation_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True)
    )
    (out / "teacher_forced.json").write_text(
        json.dumps(jsonable(teacher_forced), indent=2, sort_keys=True, allow_nan=False)
    )
    (out / "aggregate_metrics.json").write_text(
        json.dumps(
            jsonable(
                {
                    "protocol": asdict(cfg),
                    "episodes": rows,
                    "aggregate": aggregate,
                    "teacher_forced": teacher_forced["aggregate"],
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    with (out / "aggregate_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["episode_index", *_SCALAR, *physics_keys]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved continuous residual evaluation to {out}")


if __name__ == "__main__":
    main()
