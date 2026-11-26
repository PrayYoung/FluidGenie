"""Generate GitHub visuals exclusively from saved prediction and simulator arrays."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from fluidgenie.eval.showcase import grid_frame, save_gif, vorticity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    record = {"sources": {}, "visuals": {}}
    paths = [
        (
            name,
            Path(a.physics) / folder / f"{key}/episode_000000/h118/rollout_arrays.npz",
        )
        for name, folder, key in [
            ("CR", "benchmark_seed0", "cr"),
            ("P2 (Sobolev)", "p2_benchmark_seed0", "cr"),
            ("FM (Heun-16)", "benchmark_seed0", "fm"),
        ]
    ]
    predictions = []
    truth = None
    for name, path in paths:
        with np.load(path) as z:
            gt, pred = z["gt"].copy(), z["pred"].copy()
        if truth is None:
            truth = gt
        else:
            np.testing.assert_array_equal(truth, gt)
        assert np.isfinite(pred).all()
        predictions.append(pred)
        record["sources"][str(path)] = digest(path)
    scale = max(float(np.quantile(np.abs(vorticity(truth)), 0.99)), 1e-8)
    indices = list(range(0, len(truth), 2))
    if indices[-1] != len(truth) - 1:
        indices.append(len(truth) - 1)
    frames = [
        grid_frame(
            [[truth[i], *[p[i] for p in predictions]]],
            ["PhiFlow truth", *[p[0] for p in paths]],
            [""],
            f"Actual saved rollouts | forecast {i+1}/{len(truth)}",
            view="vorticity",
            limit=scale,
            size=160,
        )
        for i in indices
    ]
    gif_info = save_gif(frames, out / "turbulence_rollout.gif")
    record["visuals"]["turbulence_rollout.gif"] = {
        **gif_info,
        "source_forecast_frames": [
            indices[i] + 1 for i in gif_info["source_frame_indices"]
        ],
        "view": "vorticity",
        "shared_symmetric_limit": scale,
        "scale_protocol": "Ground-truth absolute vorticity 99th percentile; values beyond the fixed range are clipped for display.",
        "selected_case": "Heldout episode0, training seed0, selected by index.",
    }
    if getattr(a, "physics_only", False):
        (out / "visuals_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record["visuals"], indent=2))
        return record
    path = Path(a.counterfactual)
    with np.load(path) as z:
        gt, pred, context = z["gt"].copy(), z["pred"].copy(), z["context"].copy()
        simulator_metadata = json.loads(str(z["simulator_metadata"]))
    assert gt.shape == pred.shape and gt.shape[0] == 5 and context.shape[0] == 2
    assert np.isfinite(gt).all() and np.isfinite(pred).all()
    record["sources"][str(path)] = digest(path)
    limit = max(float(np.max(gt[..., 2])), float(np.max(pred[..., 2])), 1e-8)
    frames = [
        grid_frame(
            [gt[:, i], pred[:, i]],
            ["No-op", "Up ^", "Down v", "Right >", "Left <"],
            ["PhiFlow\ntruth", "FM + LAM"],
            f"Shared-context counterfactual | frame {i+1}/{gt.shape[1]}",
            view="density",
            limit=limit,
            size=128,
        )
        for i in range(gt.shape[1])
    ]
    record["visuals"]["counterfactual_steering.gif"] = {
        **save_gif(frames, out / "counterfactual_steering.gif"),
        "view": "smoke_density",
        "shared_vmax": limit,
        "scale_protocol": "Joint maximum across all truth/prediction branches and frames.",
        "selected_seed": simulator_metadata["seed"],
        "context_sha256": hashlib.sha256(context.tobytes()).hexdigest(),
    }
    frames[-1].save(out / "counterfactual_keyframes.png")
    square = out / "square_steering.json"
    if square.exists():
        record["visuals"]["square_steering.gif"] = json.loads(square.read_text())
    (out / "visuals_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["visuals"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--physics", default="outputs/repair_v4")
    p.add_argument(
        "--counterfactual",
        default="outputs/lam_v6/counterfactual/counterfactual_arrays.npz",
    )
    p.add_argument("--out", default="assets")
    p.add_argument("--physics-only", action="store_true")
    build(p.parse_args())
