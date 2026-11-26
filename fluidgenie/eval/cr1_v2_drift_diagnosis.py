"""Evaluation-only diagnostic for continuous residual long-horizon drift (H40 -> H80)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from configs.config_io import load_config_json
from fluidgenie.eval.metrics import jsonable
from fluidgenie.eval.flow_matching_rollout import _denorm, _load_stats, _norm
from fluidgenie.eval.continuous_residual_rollout import _load_residual_model


@dataclass
class DriftDiagnosisConfig:
    data_dir: str = "data/raw/ns2d_verified"
    stats_path: str = ""
    dyn_ckpt: str = "campaign_inputs/cr1_v2_seed0/latest"
    out_dir: str = "outputs/cr1_v2_drift_diagnosis"
    episode_indices: Tuple[int, ...] = tuple(range(13))  # 0 through 12
    start: int = 0
    context: int = 2
    horizon: int = 118
    sat_threshold: float = 0.90


def compute_vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # 2D vorticity = dv/dx - du/dy
    # grid coordinates: axis 0 is y (rows), axis 1 is x (cols)
    dv_dx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) * (v.shape[1] / 2)
    du_dy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) * (u.shape[0] / 2)
    return dv_dx - du_dy


def run_drift_diagnosis(cfg: DriftDiagnosisConfig) -> dict[str, Any]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from fluidgenie.eval.utils import resolve_stats_path

    cfg.stats_path = resolve_stats_path(cfg.stats_path, cfg.dyn_ckpt)
    if not cfg.stats_path:
        raise ValueError(
            "Drift diagnosis requires checkpoint-local or explicit normalization statistics"
        )
    stats = _load_stats(cfg.stats_path)
    raw_stats = np.load(cfg.stats_path)
    have_bounds = "min" in raw_stats and "max" in raw_stats
    train_u_min, train_v_min, train_d_min = raw_stats.get(
        "min", np.full(3, -np.inf)
    ).tolist()
    train_u_max, train_v_max, train_d_max = raw_stats.get(
        "max", np.full(3, np.inf)
    ).tolist()

    first_ep = np.load(
        Path(cfg.data_dir) / f"episode_{cfg.episode_indices[0]:06d}.npy", mmap_mode="r"
    )
    _, h, w, channels = first_ep.shape

    model, params, saved_cfg = _load_residual_model(cfg.dyn_ckpt, h, w, channels)
    from fluidgenie.data.splits import assert_checkpoint_evaluation

    assert_checkpoint_evaluation(saved_cfg["config"], cfg.data_dir)
    if (
        cfg.context != int(saved_cfg["config"]["context"])
        or cfg.start < 0
        or cfg.horizon < 1
        or channels < 3
    ):
        raise ValueError("Invalid drift context, bounds, or channel count")
    step_fn = jax.jit(
        lambda ctx: ctx[:, -1] + model.apply({"params": params}, ctx, training=False)
    )

    residual_scales = tuple(saved_cfg["config"].get("residual_scales", ()))
    if not residual_scales:
        raise ValueError("CR1-v2 checkpoint must have residual_scales")
    s_u, s_v, s_d = residual_scales

    all_episode_records = {}
    csv_rows = []

    for ep_idx in cfg.episode_indices:
        ep_path = Path(cfg.data_dir) / f"episode_{ep_idx:06d}.npy"
        fields = np.asarray(np.load(ep_path, mmap_mode="r"), dtype=np.float32)

        if cfg.start + cfg.context + cfg.horizon > len(fields):
            raise ValueError("Episode is shorter than the requested drift horizon")
        # Context frames
        ctx_norm = _norm(fields[cfg.start : cfg.start + cfg.context], stats)
        ctx = jnp.asarray(ctx_norm[None], dtype=jnp.float32)

        ep_records = []
        gt_motion_cum = 0.0
        pred_motion_cum = 0.0

        for step in range(1, cfg.horizon + 1):
            gt_t = fields[cfg.start + cfg.context + step - 1]
            gt_prev = (
                fields[cfg.start + cfg.context + step - 2]
                if step > 1
                else fields[cfg.start + cfg.context - 1]
            )

            # Model prediction
            prev_norm = np.asarray(ctx[0, -1])
            next_norm = np.asarray(step_fn(ctx)[0])
            delta_norm = next_norm - prev_norm

            # Advance context
            ctx = jnp.concatenate([ctx[:, 1:], next_norm[None, None]], axis=1)

            pred_t = _denorm(next_norm[None], stats)[0]
            pred_prev = _denorm(prev_norm[None], stats)[0]

            # Motion metrics
            gt_motion = float(np.mean(np.abs(gt_t - gt_prev)))
            pred_motion = float(np.mean(np.abs(pred_t - pred_prev)))
            gt_motion_cum += gt_motion
            pred_motion_cum += pred_motion
            inst_mr = pred_motion / (gt_motion + 1e-12)
            cum_mr = pred_motion_cum / (gt_motion_cum + 1e-12)

            # Error metrics
            frame_mse = float(np.mean((pred_t - gt_t) ** 2))
            frame_mae = float(np.mean(np.abs(pred_t - gt_t)))

            # Normalized state metrics
            state_rms = float(np.sqrt(np.mean(next_norm**2)))
            state_min = float(np.min(next_norm))
            state_max = float(np.max(next_norm))
            u_norm_rms = float(np.sqrt(np.mean(next_norm[..., 0] ** 2)))
            v_norm_rms = float(np.sqrt(np.mean(next_norm[..., 1] ** 2)))
            d_norm_rms = float(np.sqrt(np.mean(next_norm[..., 2] ** 2)))

            # Residual metrics
            res_rms = float(np.sqrt(np.mean(delta_norm**2)))
            res_u_rms = float(np.sqrt(np.mean(delta_norm[..., 0] ** 2)))
            res_v_rms = float(np.sqrt(np.mean(delta_norm[..., 1] ** 2)))
            res_d_rms = float(np.sqrt(np.mean(delta_norm[..., 2] ** 2)))
            res_state_ratio = res_rms / (state_rms + 1e-8)

            # Tanh saturation fraction (|delta_c| / s_c > threshold)
            sat_u_frac = float(
                np.mean(np.abs(delta_norm[..., 0]) / s_u >= cfg.sat_threshold)
            )
            sat_v_frac = float(
                np.mean(np.abs(delta_norm[..., 1]) / s_v >= cfg.sat_threshold)
            )
            sat_d_frac = float(
                np.mean(np.abs(delta_norm[..., 2]) / s_d >= cfg.sat_threshold)
            )
            sat_total_frac = (sat_u_frac + sat_v_frac + sat_d_frac) / 3.0

            # Physical density metrics
            d_pred = pred_t[..., 2]
            d_gt = gt_t[..., 2]
            pred_density_mass = float(np.sum(d_pred))
            gt_density_mass = float(np.sum(d_gt))
            pred_density_mean = float(np.mean(d_pred))
            gt_density_mean = float(np.mean(d_gt))
            pred_density_min = float(np.min(d_pred))
            pred_density_max = float(np.max(d_pred))
            d_out_of_bounds = (
                float(pred_density_min < train_d_min or pred_density_max > train_d_max)
                if have_bounds
                else None
            )

            # Velocity magnitude & kinetic energy
            u_pred, v_pred = pred_t[..., 0], pred_t[..., 1]
            u_gt, v_gt = gt_t[..., 0], gt_t[..., 1]
            vmag_pred = np.sqrt(u_pred**2 + v_pred**2)
            vmag_gt = np.sqrt(u_gt**2 + v_gt**2)
            ke_pred = float(0.5 * np.mean(vmag_pred**2))
            ke_gt = float(0.5 * np.mean(vmag_gt**2))
            pred_vmag_mean = float(np.mean(vmag_pred))
            pred_vmag_max = float(np.max(vmag_pred))
            pred_velocity_rms = float(np.sqrt(np.mean(u_pred**2 + v_pred**2)))
            gt_vmag_mean = float(np.mean(vmag_gt))
            gt_velocity_rms = float(np.sqrt(np.mean(u_gt**2 + v_gt**2)))
            gt_vmag_max = float(np.max(vmag_gt))
            u_out_of_bounds = float(
                np.min(u_pred) < train_u_min or np.max(u_pred) > train_u_max
            )
            v_out_of_bounds = float(
                np.min(v_pred) < train_v_min or np.max(v_pred) > train_v_max
            )
            vel_out_of_bounds = (
                float(u_out_of_bounds or v_out_of_bounds) if have_bounds else None
            )

            # Vorticity & enstrophy
            vort_pred = compute_vorticity(u_pred, v_pred)
            vort_gt = compute_vorticity(u_gt, v_gt)
            enstrophy_pred = float(0.5 * np.mean(vort_pred**2))
            enstrophy_gt = float(0.5 * np.mean(vort_gt**2))
            pred_vort_rms = float(np.sqrt(np.mean(vort_pred**2)))
            gt_vort_rms = float(np.sqrt(np.mean(vort_gt**2)))

            rec = {
                "episode": ep_idx,
                "step": step,
                "frame_mse": frame_mse,
                "frame_mae": frame_mae,
                "gt_motion": gt_motion,
                "pred_motion": pred_motion,
                "inst_motion_ratio": inst_mr,
                "cum_motion_ratio": cum_mr,
                "state_norm_rms": state_rms,
                "state_norm_min": state_min,
                "state_norm_max": state_max,
                "u_norm_rms": u_norm_rms,
                "v_norm_rms": v_norm_rms,
                "d_norm_rms": d_norm_rms,
                "res_norm_rms": res_rms,
                "res_u_rms": res_u_rms,
                "res_v_rms": res_v_rms,
                "res_d_rms": res_d_rms,
                "res_state_ratio": res_state_ratio,
                "sat_u_frac": sat_u_frac,
                "sat_v_frac": sat_v_frac,
                "sat_d_frac": sat_d_frac,
                "sat_total_frac": sat_total_frac,
                "pred_density_mass": pred_density_mass,
                "gt_density_mass": gt_density_mass,
                "density_mass_ratio": pred_density_mass / (gt_density_mass + 1e-12),
                "pred_density_mean": pred_density_mean,
                "gt_density_mean": gt_density_mean,
                "pred_density_min": pred_density_min,
                "pred_density_max": pred_density_max,
                "density_out_of_bounds": d_out_of_bounds,
                "pred_ke": ke_pred,
                "gt_ke": ke_gt,
                "ke_ratio": ke_pred / (ke_gt + 1e-12),
                "pred_vmag_mean": pred_vmag_mean,
                "pred_vmag_max": pred_vmag_max,
                "gt_vmag_mean": gt_vmag_mean,
                "pred_velocity_rms": pred_velocity_rms,
                "gt_velocity_rms": gt_velocity_rms,
                "vel_out_of_bounds": vel_out_of_bounds,
                "pred_enstrophy": enstrophy_pred,
                "gt_enstrophy": enstrophy_gt,
                "enstrophy_ratio": enstrophy_pred / (enstrophy_gt + 1e-12),
                "pred_vort_rms": pred_vort_rms,
                "gt_vort_rms": gt_vort_rms,
            }
            ep_records.append(rec)
            csv_rows.append(rec)

        all_episode_records[ep_idx] = ep_records

    # Write CSV
    csv_path = out_dir / "drift_trace.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # Compute step-averaged trajectories
    step_means = {}
    for step in range(1, cfg.horizon + 1):
        step_recs = [all_episode_records[ep][step - 1] for ep in cfg.episode_indices]
        step_means[step] = {
            k: float(np.mean([r[k] for r in step_recs]))
            if step_recs[0][k] is not None
            else None
            for k in step_recs[0].keys()
            if k not in ("episode", "step")
        }

    def first_step_matching(predicate) -> int | None:
        return next(
            (step for step in range(1, cfg.horizon + 1) if predicate(step)), None
        )

    residual_reference = min(
        step_means[step]["res_norm_rms"]
        for step in range(min(10, cfg.horizon), min(30, cfg.horizon) + 1)
    )
    energy_onset = next(
        (
            step
            for step in range(1, cfg.horizon - 1)
            if all(
                step_means[offset]["ke_ratio"] > 2.0
                for offset in (step, step + 1, step + 2)
            )
        ),
        None,
    )
    summary = {
        "config": asdict(cfg),
        "residual_scales": {"u": s_u, "v": s_v, "density": s_d},
        "training_bounds": {
            "u": [train_u_min, train_u_max] if have_bounds else None,
            "v": [train_v_min, train_v_max] if have_bounds else None,
            "density": [train_d_min, train_d_max] if have_bounds else None,
        },
        "step_means": step_means,
        "early_warning": {
            str(step): step_means[step]
            for step in (30, 40, 50, 60, 80)
            if step <= cfg.horizon
        },
        "onsets": {
            "energy_decoupling_ke_ratio_gt_2_for_3_steps": energy_onset,
            "residual_rms_gt_2x_min_steps_10_30": first_step_matching(
                lambda step: step_means[step]["res_norm_rms"] > 2.0 * residual_reference
            ),
            "saturation_fraction_ge_0.01": first_step_matching(
                lambda step: step_means[step]["sat_total_frac"] >= 0.01
            ),
        },
    }
    (out_dir / "drift_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, allow_nan=False)
    )
    print(f"Drift diagnosis written to {out_dir}")
    return summary


if __name__ == "__main__":
    run_drift_diagnosis(tyro.cli(DriftDiagnosisConfig))
