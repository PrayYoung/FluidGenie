"""Frozen multi-episode evaluation protocol for the FM0 baseline."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tyro

from fluidgenie.eval.flow_matching_rollout import (
    FlowMatchingRolloutConfig,
    run_flow_matching_rollout,
)
from fluidgenie.eval.utils import ensure_dir


@dataclass
class FM0EvaluationConfig:
    data_dir: str
    dyn_ckpt: str
    out_dir: str
    episode_indices: Tuple[int, int, int] = (0, 1, 2)
    start: int = 0
    horizon: int = 40
    context: int = 2
    ode_steps: int = 8
    ode_sampler: str = "heun"
    stats_path: str = ""
    base_seed: int = 1000
    action_mode: str = "oracle"
    action_id: int = 0
    lam_rollout_mode: str = "context"


_SCALAR_METRICS = (
    "mse_mean",
    "mae_mean",
    "view_mae_mean",
    "gt_motion_mean",
    "pred_motion_mean",
    "motion_ratio_mean",
    "frozen_fraction",
)


def main() -> None:
    cfg = tyro.cli(FM0EvaluationConfig)
    if cfg.ode_sampler != "heun":
        raise ValueError("The frozen FM0 protocol uses --ode-sampler heun")
    if cfg.horizon != 40 or cfg.ode_steps != 8 or cfg.context != 2:
        raise ValueError(
            "The frozen FM0 protocol requires context=2, horizon=40, ode_steps=8"
        )

    out = ensure_dir(Path(cfg.out_dir))
    episode_results = []
    for episode_index in cfg.episode_indices:
        episode_name = f"episode_{episode_index:06d}.npy"
        episode_path = Path(cfg.data_dir) / episode_name
        if not episode_path.exists():
            raise FileNotFoundError(episode_path)
        seed = cfg.base_seed + episode_index
        metrics = run_flow_matching_rollout(
            FlowMatchingRolloutConfig(
                npy_path=str(episode_path),
                dyn_ckpt=cfg.dyn_ckpt,
                out_dir=str(out / f"episode_{episode_index:06d}"),
                start=cfg.start,
                horizon=cfg.horizon,
                context=cfg.context,
                ode_steps=cfg.ode_steps,
                ode_sampler=cfg.ode_sampler,
                stats_path=cfg.stats_path,
                seed=seed,
                action_mode=cfg.action_mode,
                action_id=cfg.action_id,
                lam_rollout_mode=cfg.lam_rollout_mode,
            )
        )
        episode_results.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                **{name: float(metrics[name]) for name in _SCALAR_METRICS},
            }
        )

    aggregate = {
        name: {
            "mean": float(np.mean([row[name] for row in episode_results])),
            "std": float(np.std([row[name] for row in episode_results])),
        }
        for name in _SCALAR_METRICS
    }
    (out / "evaluation_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True)
    )
    (out / "aggregate_metrics.json").write_text(
        json.dumps(
            {
                "protocol": asdict(cfg),
                "episodes": episode_results,
                "aggregate": aggregate,
            },
            indent=2,
            sort_keys=True,
        )
    )
    with (out / "aggregate_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["episode_index", "seed", *_SCALAR_METRICS]
        )
        writer.writeheader()
        writer.writerows(episode_results)
    print(f"Saved frozen FM0 evaluation artifacts to {out}")


if __name__ == "__main__":
    main()
