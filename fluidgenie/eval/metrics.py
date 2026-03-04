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

    return {
        "mse": mse,
        "mae": mae,
        "view_mae": view_mae,
        "mse_mean": float(mse.mean()),
        "mae_mean": float(mae.mean()),
        "view_mae_mean": float(view_mae.mean()),
    }
