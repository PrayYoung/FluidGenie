"""Measure simulator velocity responses for the five labeled body-force actions."""

from fluidgenie.data.dataset_npz import load_episode_actions

from pathlib import Path
import json
import argparse
import numpy as np


def calibrate(directory):
    responses = [[] for _ in range(5)]
    for path in sorted(Path(directory).glob("episode_*.npy")):
        if path.name.endswith(".actions.npy"):
            continue
        fields = np.load(path, mmap_mode="r")
        labels = load_episode_actions(path)
        increments = fields[1:, ..., :2] - fields[:-1, ..., :2]
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta.get("action_mode") == "joystick5_localized":
            n = fields.shape[1]
            coordinates = (np.arange(n) + 0.5) / n
            distance = np.minimum(
                np.abs(coordinates - 0.5), 1 - np.abs(coordinates - 0.5)
            )
            weights = np.exp(
                -(distance[:, None] ** 2 + distance[None, :] ** 2)
                / (2 * meta.get("action_radius", 0.14) ** 2)
            )
            delta = (increments * weights[None, ..., None]).sum((1, 2)) / weights.sum()
        else:
            delta = increments.mean((1, 2))
        for label, response in zip(labels[:-1], delta):
            responses[int(label)].append(response)
    if any(not x for x in responses):
        raise ValueError("All five actions must appear in calibration data")
    means = np.asarray([np.mean(x, axis=0) for x in responses])
    relative = means - means[0]
    vectors = np.asarray([(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)])
    align = [
        float(np.dot(relative[i], vectors[i]) / max(np.linalg.norm(relative[i]), 1e-12))
        for i in range(1, 5)
    ]
    return {
        "action_names": ["noop", "up", "down", "right", "left"],
        "counts": [len(x) for x in responses],
        "mean_velocity_increment": means.tolist(),
        "response_relative_to_noop": relative.tolist(),
        "direction_cosines": align,
        "calibration_gate": bool(min(align) > 0.8),
        "source": "training split only; held-out trajectories were not used to calibrate labels",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()
    report = calibrate(a.data)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2) + "\n")
    if not report["calibration_gate"]:
        raise SystemExit("Joystick calibration failed")
