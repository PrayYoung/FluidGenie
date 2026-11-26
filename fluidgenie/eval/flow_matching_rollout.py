"""Evaluate the isolated FM0 direct state-space flow-matching baseline."""

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
from fluidgenie.eval.metrics import (
    compute_rollout_metrics,
    fluid_physics_metrics,
    jsonable,
    episode_physics_metadata,
)
from fluidgenie.data.splits import assert_checkpoint_evaluation
from fluidgenie.eval.rollout_runner import visualize_rollout
from fluidgenie.eval.samplers.flow_matching import (
    sample_flow_matching_euler,
    sample_flow_matching_heun,
)
from fluidgenie.eval.utils import ensure_dir, resolve_stats_path
from fluidgenie.models.lam_conditioning import load_frozen_lam
from fluidgenie.models.flow_matching import ConditionalFlowUNet
from fluidgenie.training.checkpoint_utils import load_params


@dataclass
class FlowMatchingRolloutConfig:
    npy_path: str
    dyn_ckpt: str
    out_dir: str
    start: int = 0
    horizon: int = 40
    context: int = 2
    ode_steps: int = 8
    ode_sampler: str = "heun"  # euler | heun
    stats_path: str = ""
    seed: int = 0
    action_id: int = 0
    action_mode: str = "oracle"  # oracle | held | none (supervised checkpoints)
    make_visualizations: bool = True
    lam_rollout_mode: str = "context"


def _load_stats(stats_path: str | None):
    if not stats_path:
        return None
    stats = np.load(stats_path)
    if "min" in stats and "max" in stats:
        lo = stats["min"].reshape(1, 1, 1, -1)
        hi = stats["max"].reshape(1, 1, 1, -1)
        return ("minmax", lo, hi - lo + 1e-6)
    return (
        "standard",
        stats["mean"].reshape(1, 1, 1, -1),
        stats["std"].reshape(1, 1, 1, -1),
    )


def _norm(x, stats):
    if stats is None:
        return x
    kind, a, b = stats
    return (x - a) / b * 2.0 - 1.0 if kind == "minmax" else (x - a) / (b + 1e-6)


def _denorm(x, stats):
    if stats is None:
        return x
    kind, a, b = stats
    return (x + 1.0) * 0.5 * b + a if kind == "minmax" else x * (b + 1e-6) + a


def _metric_json(metrics: dict) -> dict:
    return {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()
    }


def run_flow_matching_rollout(cfg: FlowMatchingRolloutConfig) -> dict:
    if cfg.start < 0 or cfg.horizon < 1:
        raise ValueError("Invalid rollout bounds")
    if cfg.lam_rollout_mode not in {"context", "oracle"}:
        raise ValueError("LAM mode must be context or oracle")
    if cfg.context != 2:
        raise ValueError("FM0 requires --context 2")
    if cfg.ode_sampler not in {"euler", "heun"}:
        raise ValueError("--ode-sampler must be euler or heun")

    fields = np.load(cfg.npy_path, mmap_mode="r")
    if fields.ndim != 4:
        raise ValueError("Expected an NPY field array shaped [T,H,W,C]")
    total, h, w, channels = fields.shape
    if cfg.start + cfg.context + cfg.horizon > total:
        raise ValueError("Episode is too short for start + context + horizon")

    saved = load_config_json(cfg.dyn_ckpt)
    if not saved or saved.get("kind") != "flow_matching_fm0":
        raise ValueError(
            "dyn_ckpt must be an FM0 checkpoint directory with config.json"
        )
    model_cfg = saved["config"]
    split = assert_checkpoint_evaluation(model_cfg, Path(cfg.npy_path).parent)
    if int(model_cfg.get("context", cfg.context)) != cfg.context:
        raise ValueError("Rollout context does not match the checkpoint config")
    hidden_dims: Tuple[int, int, int] = tuple(model_cfg["hidden_dims"])
    model = ConditionalFlowUNet(
        channels=channels,
        context_frames=cfg.context,
        hidden_dims=hidden_dims,
        time_dim=int(model_cfg["time_dim"]),
        num_res_blocks=int(model_cfg["num_res_blocks"]),
        modern=bool(model_cfg.get("modern", False)),
        action_dim=int(model_cfg.get("action_dim", 0)),
    )
    stats = _load_stats(cfg.stats_path or resolve_stats_path("", cfg.dyn_ckpt))
    initial_context = jnp.asarray(
        _norm(
            np.asarray(fields[cfg.start : cfg.start + cfg.context], dtype=np.float32),
            stats,
        )[None],
        dtype=jnp.float32,
    )
    action_cond = None
    lam_encode = None
    if model_cfg.get("use_lam"):
        if cfg.action_mode == "held":
            raise ValueError(
                "Held supervised actions conflict with latent conditioning; use the calibrated steering benchmark"
            )
        if cfg.lam_rollout_mode not in {"context", "oracle"}:
            raise ValueError("LAM mode must be context or oracle")
        _, _, lam_encode, _ = load_frozen_lam(model_cfg["lam_ckpt"], initial_context)
        action_cond = lam_encode(initial_context)
    elif model.action_dim:
        if cfg.action_mode not in {"oracle", "held"}:
            raise ValueError(
                "Conditional checkpoint requires action_mode=oracle or held"
            )
        if cfg.action_id not in range(model.action_dim):
            raise ValueError("Invalid held action")
        action_cond = jax.nn.one_hot(
            jnp.asarray([cfg.action_id]), model.action_dim, dtype=jnp.float32
        )
    labels = None
    if model.action_dim and lam_encode is None and cfg.action_mode == "oracle":
        from fluidgenie.data.dataset_npz import load_episode_actions

        labels = load_episode_actions(cfg.npy_path)
        if (
            len(labels) < cfg.start + cfg.context + cfg.horizon - 1
            or np.any(labels < 0)
            or np.any(labels >= model.action_dim)
        ):
            raise ValueError("Invalid or insufficient oracle action labels")
    init_key = jax.random.PRNGKey(cfg.seed)
    params_init = model.init(
        init_key,
        jnp.zeros((1, h, w, channels), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.float32),
        jnp.zeros((1, cfg.context, h, w, channels), dtype=jnp.float32),
        training=False,
        **({"action_cond": action_cond} if action_cond is not None else {}),
    )["params"]
    params = load_params(cfg.dyn_ckpt, params_init)

    stats = _load_stats(cfg.stats_path or resolve_stats_path("", cfg.dyn_ckpt))
    normalized = _norm(
        np.asarray(fields[cfg.start : cfg.start + cfg.context], dtype=np.float32), stats
    )
    context = jnp.asarray(normalized[None], dtype=jnp.float32)
    sampler = (
        sample_flow_matching_heun
        if cfg.ode_sampler == "heun"
        else sample_flow_matching_euler
    )
    sample_once = jax.jit(
        lambda ctx, key, condition: sampler(
            model, params, ctx, key, cfg.ode_steps, action_cond=condition
        )
    )

    rng = jax.random.PRNGKey(cfg.seed + 1)
    predictions = []
    for rollout_index in range(cfg.horizon):
        if labels is not None:
            action_cond = jax.nn.one_hot(
                jnp.asarray([labels[cfg.start + cfg.context - 1 + rollout_index]]),
                model.action_dim,
                dtype=jnp.float32,
            )
        if lam_encode is not None:
            observed = context
            if cfg.lam_rollout_mode == "oracle":
                end = cfg.start + cfg.context + rollout_index + 1
                observed = jnp.asarray(
                    _norm(
                        np.asarray(
                            fields[end - cfg.context - 1 : end], dtype=np.float32
                        ),
                        stats,
                    )[None],
                    dtype=jnp.float32,
                )
            action_cond = lam_encode(observed)
        rng, step_key = jax.random.split(rng)
        next_norm = sample_once(context, step_key, action_cond)
        next_field = _denorm(np.asarray(next_norm[0])[None], stats)[0]
        predictions.append(next_field)
        context = jnp.concatenate([context[:, 1:], next_norm[:, None]], axis=1)

    pred = np.stack(predictions)
    gt = np.asarray(
        fields[cfg.start + cfg.context : cfg.start + cfg.context + cfg.horizon]
    )
    metrics = compute_rollout_metrics(gt, pred, view="density")
    out = ensure_dir(Path(cfg.out_dir))
    np.savez_compressed(out / "rollout_arrays.npz", gt=gt, pred=pred)
    (out / "physics_metrics.json").write_text(
        json.dumps(
            jsonable(
                fluid_physics_metrics(
                    gt,
                    pred,
                    last_context=fields[cfg.start + cfg.context - 1],
                    **episode_physics_metadata(cfg.npy_path),
                )
            ),
            indent=2,
        )
    )
    (out / "split_validation.json").write_text(json.dumps(split, indent=2))
    (out / "metrics.json").write_text(
        json.dumps(jsonable(metrics), indent=2, sort_keys=True, allow_nan=False)
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
        for frame, values in enumerate(
            zip(
                metrics["mse"],
                metrics["mae"],
                metrics["view_mae"],
                metrics["gt_motion"],
                metrics["pred_motion"],
                metrics["motion_ratio"],
                metrics["is_frozen"],
            )
        ):
            mse, mae, density_mae, gt_motion, pred_motion, motion_ratio, is_frozen = (
                values
            )
            writer.writerow(
                {
                    "frame": frame,
                    "mse": float(mse),
                    "mae": float(mae),
                    "density_mae": float(density_mae),
                    "gt_motion": float(gt_motion),
                    "pred_motion": float(pred_motion),
                    "motion_ratio": float(motion_ratio),
                    "is_frozen": bool(is_frozen),
                }
            )
    (out / "rollout_config.json").write_text(
        json.dumps(
            {
                **asdict(cfg),
                "sampler_protocol": "heun_interior_euler_final_v2"
                if cfg.ode_sampler == "heun"
                else "euler",
            },
            indent=2,
            sort_keys=True,
        )
    )
    if cfg.make_visualizations:
        for view in ("density", "speed", "vorticity"):
            visualize_rollout(gt, pred, ensure_dir(out / view), view=view)
    print(f"Saved FM0 rollout artifacts to {out}")
    return metrics


def main() -> None:
    cfg = tyro.cli(FlowMatchingRolloutConfig)
    run_flow_matching_rollout(cfg)


if __name__ == "__main__":
    main()
