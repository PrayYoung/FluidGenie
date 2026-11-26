from __future__ import annotations

import numpy as np
from jaxtyping import Array, Float

from fluidgenie.eval.utils import vorticity_from_uv


def compute_rollout_metrics(
    gt: Float[Array, "t h w c"], pred: Float[Array, "t h w c"], view: str
) -> dict:
    """
    gt, pred: [T,H,W,C]
    Returns per-frame metrics and aggregates.
    """
    assert gt.shape == pred.shape
    T, _, _, C = gt.shape
    mse = np.mean((gt - pred) ** 2, axis=(1, 2, 3))
    mae = np.mean(np.abs(gt - pred), axis=(1, 2, 3))

    if view == "density" and C >= 3:
        gt_v = gt[..., 2]
        pr_v = pred[..., 2]
    elif view == "speed" and C >= 2:
        gt_v = np.sqrt(gt[..., 0] ** 2 + gt[..., 1] ** 2)
        pr_v = np.sqrt(pred[..., 0] ** 2 + pred[..., 1] ** 2)
    elif view == "vorticity" and C >= 2:
        gt_v = np.stack([vorticity_from_uv(gt[t, ..., :2]) for t in range(T)], axis=0)
        pr_v = np.stack([vorticity_from_uv(pred[t, ..., :2]) for t in range(T)], axis=0)
    else:
        gt_v = gt[..., 0]
        pr_v = pred[..., 0]

    view_mae = np.mean(np.abs(pr_v - gt_v), axis=(1, 2))

    # Motion diagnostics intentionally use the full physical state, not only the
    # selected visualization channel. Frame zero has no preceding transition.
    gt_motion = np.zeros(T, dtype=np.float64)
    pred_motion = np.zeros(T, dtype=np.float64)
    if T > 1:
        gt_motion[1:] = np.mean(np.abs(gt[1:] - gt[:-1]), axis=(1, 2, 3))
        pred_motion[1:] = np.mean(np.abs(pred[1:] - pred[:-1]), axis=(1, 2, 3))
    motion_ratio = np.divide(
        pred_motion,
        gt_motion,
        out=np.ones_like(pred_motion),
        where=gt_motion > 1e-8,
    )
    # A frame is static relative to a demonstrably moving ground-truth frame.
    is_frozen = (gt_motion > 1e-8) & (pred_motion < 0.1 * gt_motion)

    valid_motion = gt_motion[1:] > 1e-8
    gt_motion_valid = gt_motion[1:][valid_motion]
    pred_motion_valid = pred_motion[1:][valid_motion]
    ratios_valid = motion_ratio[1:][valid_motion]
    if ratios_valid.size:
        motion_ratio_of_means = float(pred_motion_valid.mean() / gt_motion_valid.mean())
        motion_ratio_median = float(np.median(ratios_valid))
        motion_ratio_mean = float(ratios_valid.mean())
    else:
        motion_ratio_of_means = 1.0
        motion_ratio_median = 1.0
        motion_ratio_mean = 1.0

    return {
        "mse": mse,
        "mae": mae,
        "view_mae": view_mae,
        "mse_mean": float(mse.mean()),
        "mae_mean": float(mae.mean()),
        "view_mae_mean": float(view_mae.mean()),
        "gt_motion": gt_motion,
        "pred_motion": pred_motion,
        "motion_ratio": motion_ratio,
        "is_frozen": is_frozen,
        "gt_motion_mean": float(gt_motion[1:].mean()) if T > 1 else 0.0,
        "pred_motion_mean": float(pred_motion[1:].mean()) if T > 1 else 0.0,
        # These intentionally distinguish ratio-of-means from ratio summaries.
        # The latter can be dominated by physically near-static GT transitions.
        "motion_ratio_of_means": motion_ratio_of_means,
        "motion_ratio_median": motion_ratio_median,
        "motion_ratio_mean": motion_ratio_mean,
        "frozen_fraction": float(is_frozen[1:].mean()) if T > 1 else 0.0,
    }


def fluid_physics_metrics(gt, pred, dx=None, dt=1.0, substeps=1, last_context=None):
    """Physical-space periodic velocity diagnostics and radial energy spectra."""
    gt, pred = np.asarray(gt, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 4 or gt.shape[-1] < 2:
        raise ValueError("Expected matching [T,H,W,C>=2] physical fields")
    h, w = gt.shape[1:3]
    dy = 1.0 / h if dx is None else dx
    dx = 1.0 / w if dx is None else dx

    def diagnostics(x):
        u, v = x[..., 0], x[..., 1]
        omega = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2 * dx) - (
            np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)
        ) / (2 * dy)
        div = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2 * dx) + (
            np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)
        ) / (2 * dy)
        tke = 0.5 * np.mean(
            (u - u.mean((1, 2), keepdims=True)) ** 2
            + (v - v.mean((1, 2), keepdims=True)) ** 2,
            axis=(1, 2),
        )
        energy = (
            0.5
            * (
                np.abs(np.fft.fft2(u, axes=(1, 2))) ** 2
                + np.abs(np.fft.fft2(v, axes=(1, 2))) ** 2
            )
            / (h * w) ** 2
        )
        ky, kx = np.meshgrid(
            np.fft.fftfreq(h) * h, np.fft.fftfreq(w) * w, indexing="ij"
        )
        bins = np.round(np.sqrt(kx * kx + ky * ky)).astype(int)
        spectra = np.stack(
            [
                np.bincount(bins.ravel(), weights=e.ravel(), minlength=bins.max() + 1)
                for e in energy
            ]
        )
        return omega, {
            "enstrophy": 0.5 * np.mean(omega**2, axis=(1, 2)),
            "tke": tke,
            "kinetic_energy": 0.5 * np.mean(u * u + v * v, axis=(1, 2)),
            "divergence_rms": np.sqrt(np.mean(div * div, axis=(1, 2))),
            "energy_spectrum": spectra,
        }

    true_omega, true = diagnostics(gt)
    pred_omega, estimate = diagnostics(pred)
    a = true_omega - true_omega.mean((1, 2), keepdims=True)
    b = pred_omega - pred_omega.mean((1, 2), keepdims=True)
    denom = np.sqrt(np.sum(a * a, axis=(1, 2)) * np.sum(b * b, axis=(1, 2)))
    corr = np.divide(
        np.sum(a * b, axis=(1, 2)), denom, out=np.zeros_like(denom), where=denom > 1e-12
    )
    flat_equal = np.all(
        np.isclose(true_omega, pred_omega, atol=1e-12, rtol=1e-6), axis=(1, 2)
    )
    corr = np.where((denom <= 1e-12) & flat_equal, 1.0, corr)
    persistence = np.mean((gt[1:] - gt[:-1]) ** 2, axis=(1, 2, 3))
    spectral_annulus = (np.arange(true["energy_spectrum"].shape[1]) >= 1) & (
        np.arange(true["energy_spectrum"].shape[1]) <= min(h, w) // 2
    )
    report = {
        "vorticity_correlation": corr,
        "ground_truth": true,
        "prediction": estimate,
        "lag1_step_mse": persistence,
        "spectrum_l1": np.mean(
            np.abs(
                true["energy_spectrum"][:, spectral_annulus]
                - estimate["energy_spectrum"][:, spectral_annulus]
            ),
            axis=1,
        ),
    }
    if dt <= 0 or substeps < 1:
        raise ValueError("Physical dt and substeps must be positive")

    def ratio(a, b):
        return np.divide(
            a, b, out=np.where(np.abs(a) <= 1e-12, 1.0, np.nan), where=np.abs(b) > 1e-12
        )

    times = (np.arange(len(gt)) + 1) * dt * substeps

    def crossing(threshold):
        indices = np.flatnonzero(corr < threshold)
        return float(times[indices[0]]) if indices.size else None

    report.update(
        physical_times=times,
        vorticity_crossing_time_below_0_8=crossing(0.8),
        vorticity_crossing_time_below_0_5=crossing(0.5),
        kinetic_energy_ratio=ratio(estimate["kinetic_energy"], true["kinetic_energy"]),
        tke_ratio=ratio(estimate["tke"], true["tke"]),
        enstrophy_ratio=ratio(estimate["enstrophy"], true["enstrophy"]),
    )
    spectrum_true, spectrum_pred = true["energy_spectrum"], estimate["energy_spectrum"]
    report["spectrum_log10_error"] = np.mean(
        np.abs(
            np.log10(spectrum_pred[:, spectral_annulus] + 1e-12)
            - np.log10(spectrum_true[:, spectral_annulus] + 1e-12)
        ),
        axis=1,
    )
    high = (np.arange(spectrum_true.shape[1]) >= min(h, w) / 4) & spectral_annulus
    report["high_k_log10_bias"] = np.mean(
        np.log10(spectrum_pred[:, high] + 1e-12)
        - np.log10(spectrum_true[:, high] + 1e-12),
        axis=1,
    )
    report["high_k_energy_ratio"] = ratio(
        spectrum_pred[:, high].sum(axis=1), spectrum_true[:, high].sum(axis=1)
    )
    if gt.shape[-1] >= 3:

        def sharpness(x):
            rho = x[..., 2]
            gx = (np.roll(rho, -1, axis=2) - np.roll(rho, 1, axis=2)) / (2 * dx)
            gy = (np.roll(rho, -1, axis=1) - np.roll(rho, 1, axis=1)) / (2 * dy)
            return np.mean(gx * gx + gy * gy, axis=(1, 2))

        report["density_grad_energy"] = sharpness(pred)
        report["smoke_sharpness_ratio"] = ratio(sharpness(pred), sharpness(gt))
    persistence_reference = (
        "last_context" if last_context is not None else "first_target"
    )
    persistence_start = 0 if last_context is not None else 1
    if last_context is None:
        last_context = gt[0]
    if last_context is not None:
        last_context = np.asarray(last_context, dtype=np.float64)
        if last_context.shape != gt.shape[1:]:
            raise ValueError(
                "last_context must be the exact final observed physical frame"
            )
        frozen = np.repeat(last_context[None], len(gt), axis=0)
        lagged = np.concatenate([last_context[None], gt[:-1]], axis=0)
        frozen_mse = np.mean((gt - frozen) ** 2, axis=(1, 2, 3))
        lagged_mse = np.mean((gt - lagged) ** 2, axis=(1, 2, 3))
        model_mse = np.mean((gt - pred) ** 2, axis=(1, 2, 3))
        report.update(
            persistence_target_indices=np.arange(persistence_start, len(gt)),
            persistence_mse=frozen_mse[persistence_start:],
            persistence_reference=persistence_reference,
            frozen_context_persistence_mse=frozen_mse[persistence_start:],
            lag1_persistence_mse=lagged_mse[persistence_start:],
            mse_skill_vs_persistence=(1 - ratio(model_mse, frozen_mse))[
                persistence_start:
            ],
        )
    return report


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, np.generic)):
        return jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def episode_physics_metadata(npy_path):
    import json
    from pathlib import Path

    path = Path(npy_path).with_suffix(".json")
    meta = json.loads(path.read_text()) if path.exists() else {}
    return {
        "dt": float(meta.get("dt", 1.0)) * int(meta.get("save_every", 1)),
        "substeps": int(meta.get("substeps", 1)),
    }


def steering_response_metrics(pred, truth, pred_noop, truth_noop, radius=0.14):
    """Compare zero-mean control response fields, rather than global mean flow."""
    pred, truth, pred_noop, truth_noop = map(
        np.asarray, (pred, truth, pred_noop, truth_noop)
    )
    if not (pred.shape == truth.shape == pred_noop.shape == truth_noop.shape):
        raise ValueError("Steering branches must have matching shapes")
    if radius <= 0:
        raise ValueError("Jet weighting radius must be positive")
    a, b = pred[..., :2] - pred_noop[..., :2], truth[..., :2] - truth_noop[..., :2]
    h, w = pred.shape[1:3]
    y, x = np.meshgrid(
        (np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij"
    )
    weights = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * radius**2))[
        None, :, :, None
    ]

    def cosine(weight):
        denominator = np.sqrt(np.sum(weight * a * a) * np.sum(weight * b * b))
        return (
            float(np.sum(weight * a * b) / denominator) if denominator > 1e-12 else None
        )

    density = None
    if pred.shape[-1] >= 3:
        rho_a, rho_b = pred[..., 2].ravel(), truth[..., 2].ravel()
        rho_a, rho_b = rho_a - rho_a.mean(), rho_b - rho_b.mean()
        denominator = np.linalg.norm(rho_a) * np.linalg.norm(rho_b)
        if denominator > 1e-12:
            density = float(np.dot(rho_a, rho_b) / denominator)
    return dict(
        velocity_delta_cosine=cosine(1.0),
        velocity_delta_cosine_localized=cosine(weights),
        density_correlation=density,
    )


def summarize_physics(physics):
    """Finite summaries with explicit null for undefined ratios or uncrossed thresholds."""
    summary = {}
    for name in (
        "vorticity_correlation",
        "tke_ratio",
        "enstrophy_ratio",
        "divergence_rms",
        "spectrum_log10_error",
        "high_k_log10_bias",
        "smoke_sharpness_ratio",
        "mse_skill_vs_persistence",
    ):
        value = physics.get(name, physics.get("prediction", {}).get(name))
        if value is None:
            summary[name] = None
            continue
        values = np.asarray(value, dtype=float)
        valid = values[np.isfinite(values)]
        summary[name] = float(valid.mean()) if valid.size else None
    for name in (
        "vorticity_crossing_time_below_0_8",
        "vorticity_crossing_time_below_0_5",
    ):
        summary[name] = physics.get(name)
    return summary
