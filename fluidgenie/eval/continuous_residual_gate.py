"""Predeclared CR1 gate; no model computation or training occurs here."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class CR1GateConfig:
    fm0_controls: str
    cr0_eval_dir: str
    out_path: str


def main() -> None:
    cfg = tyro.cli(CR1GateConfig)
    fm0 = json.loads(Path(cfg.fm0_controls).read_text())["one_step_conditional_mean"]["1"]["aggregate"]
    cr0 = json.loads((Path(cfg.cr0_eval_dir) / "teacher_forced.json").read_text())["aggregate"]
    first, terminal = [], []
    for metrics_path in sorted(Path(cfg.cr0_eval_dir).glob("episode_*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        first.append(float(metrics["mse"][0]))
        terminal.append(float(metrics["mse"][-1]))
    first_mse, terminal_mse = float(np.mean(first)), float(np.mean(terminal))
    calibration_pass = (
        cr0["mse"] <= fm0["mse"]
        and cr0["motion_ratio_of_means"] <= 0.5 * fm0["motion_ratio_of_means"]
    )
    drift_present = terminal_mse >= 1.5 * first_mse and terminal_mse - first_mse >= 0.01
    result = {
        "criteria": {
            "calibration": "CR0 teacher-forced MSE <= FM0 K=1 MSE and CR0 ratio-of-means <= 0.5 * FM0 K=1 ratio-of-means",
            "drift": "mean horizon-40 MSE >= 1.5 * mean horizon-1 MSE and increase >= 0.01",
        },
        "fm0_k1": fm0,
        "cr0_teacher_forced": cr0,
        "cr0_first_frame_mse": first_mse,
        "cr0_horizon40_mse": terminal_mse,
        "calibration_pass": calibration_pass,
        "drift_present": drift_present,
        "run_cr1": bool(calibration_pass and drift_present),
    }
    out = Path(cfg.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
