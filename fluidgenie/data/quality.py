"""Finite-state, motion, and finite-window stationarity diagnostics for fluid episodes."""

from pathlib import Path
import json
import numpy as np


def validate(data):
    rows = []
    for p in sorted(Path(data).glob("*.npy")):
        if p.name.endswith(".actions.npy"):
            continue
        x = np.load(p, mmap_mode="r")
        if x.ndim != 4 or x.shape[-1] < 2 or len(x) < 4:
            raise ValueError("Invalid episode shape")
        ke = 0.5 * np.mean(x[..., :2] ** 2, axis=(1, 2, 3)) * 2
        q = max(1, len(ke) // 4)
        ratio = float(ke[-q:].mean() / max(ke[:q].mean(), 1e-12))
        transitions = np.diff(x, axis=0)
        motion = float(np.mean(np.abs(transitions)))
        frame_motion = np.sqrt(np.mean(transitions**2, axis=(1, 2, 3)))
        motion_ratio = float(
            frame_motion[-q:].mean() / max(frame_motion[:q].mean(), 1e-12)
        )
        meta_path = p.with_suffix(".json")
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        samples = []
        dx = 1 / x.shape[2]
        dy = 1 / x.shape[1]
        for t in (0, 20, 40, 80, 120, 160):
            if t >= len(x):
                continue
            u, v = x[t, ..., 0], x[t, ..., 1]
            du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
            dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
            omega = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx) - (
                np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)
            ) / (2 * dy)
            samples.append(
                {
                    "frame": t,
                    "time_since_burn_in": t
                    * metadata.get("dt", 1)
                    * metadata.get("substeps", 1)
                    * metadata.get("save_every", 1),
                    "u_rms": float(np.sqrt(np.mean(u * u))),
                    "vorticity_rms": float(np.sqrt(np.mean(omega * omega))),
                    "gt_motion_rms": float(frame_motion[t])
                    if t < len(frame_motion)
                    else None,
                    "max_abs_divergence": float(np.max(np.abs(du_dx + dv_dy))),
                }
            )
        row = {
            "episode": p.name,
            "finite": bool(np.isfinite(x).all()),
            "kinetic_energy_ratio_last_first_quarter": ratio,
            "mean_motion": motion,
            "motion_rms_ratio_last_first_quarter": motion_ratio,
            "sampled_diagnostics": samples,
            "density_std": float(x[..., 2].std()) if x.shape[-1] > 2 else None,
            "stationarity_gate": bool(
                np.isfinite(motion)
                and np.isfinite(ratio)
                and np.isfinite(motion_ratio)
                and 0.5 <= ratio <= 2.0
                and 0.5 <= motion_ratio <= 2.0
                and motion > 1e-3
            ),
        }
        rows.append(row)
    if not rows:
        raise ValueError("No episodes")
    return {
        "episodes": rows,
        "finite_gate": all(x["finite"] for x in rows),
        "persistent_motion_gate": all(
            np.isfinite(x["mean_motion"]) and x["mean_motion"] > 1e-3 for x in rows
        ),
        "stationarity_gate": all(x["stationarity_gate"] for x in rows),
        "stationarity_definition": "Kinetic energy and RMS interframe motion means over last/first quarter in [0.5,2], with mean absolute interframe motion >1e-3. This is a finite-window gate, not proof of statistical stationarity.",
    }
