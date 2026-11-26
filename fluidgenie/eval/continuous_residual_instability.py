"""Evaluation-only, stepwise CR1 rollout-instability trace."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from fluidgenie.eval.continuous_residual_rollout import _load_residual_model
from fluidgenie.eval.flow_matching_rollout import _denorm, _load_stats, _norm


@dataclass
class CR1InstabilityConfig:
    data_dir: str
    dyn_ckpt: str
    stats_path: str
    out_dir: str
    episode_indices: Tuple[int, ...] = (0, 1, 2)
    start: int = 0
    horizon: int = 40
    context: int = 2
    state_abs_threshold_normalized: float = 50.0
    residual_growth_threshold: float = 5.0
    residual_rms_floor: float = 1e-5
    make_plots: bool = True


_CHANNELS = ("u", "v", "density")


def _stats(x: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    per_channel = {}
    for channel, name in enumerate(_CHANNELS):
        values = x[..., channel]
        per_channel[name] = {
            "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
            "std": float(values.std()), "rms": float(np.sqrt(np.mean(values ** 2))),
            "l2": float(np.linalg.norm(values.ravel())),
        }
    return {
        "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std()),
        "rms": float(np.sqrt(np.mean(x ** 2))), "l2": float(np.linalg.norm(x.ravel())),
        "max_abs": float(np.abs(x).max()), "finite": bool(np.isfinite(x).all()), "per_channel": per_channel,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode_index", "step", "frame_mse_physical", "frame_mae_physical", "frame_mse_normalized",
        "gt_motion_physical", "pred_motion_physical", "motion_ratio", "state_norm_rms", "state_phys_rms",
        "delta_norm_rms", "delta_phys_rms", "delta_state_norm_ratio", "residual_growth_factor",
        "state_norm_max_abs", "first_nonfinite", "sudden_residual_amplification",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fields})


def _training_summary(ckpt: str) -> dict[str, Any]:
    root = Path(ckpt).resolve().parent
    log_path = root / "continuous_residual_metrics.jsonl"
    if not log_path.exists():
        return {"available": False, "reason": "continuous_residual_metrics.jsonl was not found"}
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    losses = [float(row["loss"]) for row in rows]
    return {
        "available": True,
        "num_logged_rows": len(rows), "first_loss": losses[0], "last_loss": losses[-1],
        "min_loss": min(losses), "max_loss": max(losses), "loss_spike_factor": max(losses) / max(min(losses), 1e-12),
        "logged_fields": sorted(rows[0]),
        "limitations": "The training log has aggregate loss and unroll_steps only; it has no gradient norms, per-unroll-step terms, episode IDs, or sample rarity information.",
    }


def _plot(out: Path, episode_rows: dict[int, list[dict[str, Any]]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = (("state_norm_rms", "Normalized state RMS"), ("delta_norm_rms", "Normalized residual RMS"),
             ("frame_mse_physical", "Physical frame MSE"))
    fig, axes = plt.subplots(len(specs), 1, figsize=(8, 8), sharex=True)
    for axis, (key, label) in zip(axes, specs):
        for episode, rows in episode_rows.items():
            axis.plot([row["step"] for row in rows], [row[key] for row in rows], label=f"episode {episode}")
        axis.set_yscale("symlog", linthresh=1e-8)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("rollout step")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "cr1_instability_curves.png", dpi=160)
    plt.close(fig)


def run_trace(cfg: CR1InstabilityConfig) -> dict[str, Any]:
    if cfg.horizon < 1:
        raise ValueError("horizon must be positive")
    if cfg.context not in (1, 2):
        raise ValueError("context must be 1 or 2")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    first = np.load(Path(cfg.data_dir) / f"episode_{cfg.episode_indices[0]:06d}.npy", mmap_mode="r")
    _, h, w, channels = first.shape
    model, params, saved = _load_residual_model(cfg.dyn_ckpt, h, w, channels)
    if saved["kind"] not in {"continuous_residual_cr0", "continuous_residual_cr1_k4", "continuous_residual_cr1_v2_k4_bounded", "continuous_residual_cr1_v3_noise", "continuous_residual_cr1_v3_k8", "continuous_residual_cr1_physics"}:
        raise ValueError("This diagnosis is restricted to a continuous residual checkpoint")
    if int(saved["config"]["context"]) != cfg.context:
        raise ValueError("trace context must match checkpoint context")
    residual_fn = jax.jit(lambda context: model.apply({"params": params}, context, training=False))
    stats = _load_stats(cfg.stats_path)
    all_rows: list[dict[str, Any]] = []
    episode_rows: dict[int, list[dict[str, Any]]] = {}
    episode_summary: list[dict[str, Any]] = []

    for episode in cfg.episode_indices:
        fields = np.asarray(np.load(Path(cfg.data_dir) / f"episode_{episode:06d}.npy", mmap_mode="r"), dtype=np.float32)
        if cfg.start + cfg.context + cfg.horizon > len(fields):
            raise ValueError("Episode too short for requested trace horizon")
        context = jnp.asarray(_norm(fields[cfg.start:cfg.start + cfg.context], stats)[None], dtype=jnp.float32)
        rows: list[dict[str, Any]] = []
        first_threshold = first_nonfinite = first_amplification = None
        previous_delta_rms = None
        for step in range(cfg.horizon):
            current_norm = np.asarray(context[0, -1])
            delta_norm = np.asarray(residual_fn(context)[0])
            next_norm = current_norm + delta_norm
            current_phys = _denorm(current_norm[None], stats)[0]
            next_phys = _denorm(next_norm[None], stats)[0]
            delta_phys = next_phys - current_phys
            target_phys = fields[cfg.start + cfg.context + step]
            target_norm = _norm(target_phys[None], stats)[0]
            gt_previous = fields[cfg.start + cfg.context - 1 + step]
            state_norm_stats, state_phys_stats = _stats(next_norm), _stats(next_phys)
            delta_norm_stats, delta_phys_stats = _stats(delta_norm), _stats(delta_phys)
            gt_motion = float(np.mean(np.abs(target_phys - gt_previous)))
            pred_motion = float(np.mean(np.abs(next_phys - current_phys)))
            growth = None if previous_delta_rms is None else delta_norm_stats["rms"] / max(previous_delta_rms, 1e-12)
            sudden = bool(growth is not None and growth >= cfg.residual_growth_threshold and delta_norm_stats["rms"] >= cfg.residual_rms_floor)
            nonfinite = not state_norm_stats["finite"] or not delta_norm_stats["finite"]
            threshold = state_norm_stats["max_abs"] >= cfg.state_abs_threshold_normalized
            if threshold and first_threshold is None:
                first_threshold = step
            if nonfinite and first_nonfinite is None:
                first_nonfinite = step
            if sudden and first_amplification is None:
                first_amplification = step
            row = {
                "episode_index": episode, "step": step,
                "state_normalized": state_norm_stats, "state_physical": state_phys_stats,
                "delta_normalized": delta_norm_stats, "delta_physical": delta_phys_stats,
                "frame_mse_physical": float(np.mean((next_phys - target_phys) ** 2)),
                "frame_mae_physical": float(np.mean(np.abs(next_phys - target_phys))),
                "frame_mse_normalized": float(np.mean((next_norm - target_norm) ** 2)),
                "gt_motion_physical": gt_motion, "pred_motion_physical": pred_motion,
                "motion_ratio": pred_motion / max(gt_motion, 1e-8),
                "state_norm_rms": state_norm_stats["rms"], "state_phys_rms": state_phys_stats["rms"],
                "delta_norm_rms": delta_norm_stats["rms"], "delta_phys_rms": delta_phys_stats["rms"],
                "delta_state_norm_ratio": delta_norm_stats["rms"] / max(_stats(current_norm)["rms"], 1e-12),
                "residual_growth_factor": growth, "state_norm_max_abs": state_norm_stats["max_abs"],
                "first_nonfinite": nonfinite, "sudden_residual_amplification": sudden,
            }
            rows.append(row); all_rows.append(row)
            previous_delta_rms = delta_norm_stats["rms"]
            context = jnp.concatenate([context[:, 1:], jnp.asarray(next_norm[None, None])], axis=1)
        ratios = np.asarray([row["motion_ratio"] for row in rows])
        gt_motion = np.asarray([row["gt_motion_physical"] for row in rows])
        pred_motion = np.asarray([row["pred_motion_physical"] for row in rows])
        episode_rows[episode] = rows
        episode_summary.append({
            "episode_index": episode, "first_threshold_step": first_threshold, "first_nonfinite_step": first_nonfinite,
            "first_sudden_residual_amplification_step": first_amplification,
            "ratio_of_mean_motions": float(pred_motion.mean() / max(gt_motion.mean(), 1e-8)),
            "median_per_transition_motion_ratio": float(np.median(ratios)),
            "mean_per_transition_motion_ratio": float(ratios.mean()),
            "first_frame_mse": rows[0]["frame_mse_physical"], "final_frame_mse": rows[-1]["frame_mse_physical"],
            "max_state_normalized_abs": max(row["state_norm_max_abs"] for row in rows),
            "max_delta_normalized_rms": max(row["delta_norm_rms"] for row in rows),
        })

    result = {"config": asdict(cfg), "checkpoint_config": saved, "training_log": _training_summary(cfg.dyn_ckpt),
              "episodes": episode_summary, "per_step": all_rows}
    (out / "cr1_instability_trace.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (out / "cr1_instability_summary.json").write_text(json.dumps({
        "config": asdict(cfg), "training_log": result["training_log"], "episodes": episode_summary,
    }, indent=2, sort_keys=True))
    _write_csv(out / "cr1_instability_trace.csv", all_rows)
    np.savez_compressed(out / "cr1_instability_core.npz",
                        episode=np.asarray([row["episode_index"] for row in all_rows]), step=np.asarray([row["step"] for row in all_rows]),
                        state_norm_rms=np.asarray([row["state_norm_rms"] for row in all_rows]), delta_norm_rms=np.asarray([row["delta_norm_rms"] for row in all_rows]),
                        frame_mse=np.asarray([row["frame_mse_physical"] for row in all_rows]), motion_ratio=np.asarray([row["motion_ratio"] for row in all_rows]))
    if cfg.make_plots:
        _plot(out, episode_rows)
    print(f"Saved CR1 instability trace to {out}")
    return result


def main() -> None:
    run_trace(tyro.cli(CR1InstabilityConfig))


if __name__ == "__main__":
    main()
