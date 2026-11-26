"""Five held-action branches from identical simulated context and model noise."""

from dataclasses import dataclass, fields as dataclass_fields
from configs.config_io import load_config_json
from pathlib import Path
import json
import tempfile
import numpy as np
import tyro
from configs.gen_data_configs import NS2DConfig
from fluidgenie.data.gen_phiflow_ns2d import generate_counterfactual_branches
from fluidgenie.cli.steering_demo import SteeringSession
from fluidgenie.eval.metrics import (
    fluid_physics_metrics,
    jsonable,
    steering_response_metrics,
    summarize_physics,
)


@dataclass
class Args:
    checkpoint: str
    stats: str
    out: str
    seed: int = 200003
    horizon: int = 40
    resolution: int = 0
    burn_in: int = -1
    ode_steps: int = 16
    data: str = ""
    seeds: tuple[int, ...] = ()


def evaluate_seed(a, seed, root, session=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = []
    truths = []
    with tempfile.TemporaryDirectory() as temp:
        initial = None
        model_cfg = load_config_json(a.checkpoint)["config"]
        train_meta = next(Path(a.data or model_cfg["data"]).glob("episode_*.json"))
        metadata = json.loads(train_meta.read_text())
        simulator = {
            field.name: metadata[field.name]
            for field in dataclass_fields(NS2DConfig)
            if field.name in metadata
        }
        cfg = NS2DConfig(
            **{
                **simulator,
                "seed": seed,
                "resolution": a.resolution or simulator.get("resolution", 64),
                "steps": a.horizon + 2,
                "burn_in": (
                    simulator.get("burn_in", 1000) if a.burn_in < 0 else a.burn_in
                ),
                "action_start": 1,
            }
        )
        branches = generate_counterfactual_branches(cfg)
        for action in range(5):
            fields, meta = branches[action]
            if initial is None:
                initial = fields[:2].copy()
            else:
                np.testing.assert_array_equal(fields[:2], initial)
            path = Path(temp) / "context.npy"
            np.save(path, initial)
            if session is None:
                session = SteeringSession(
                    a.checkpoint, str(path), a.stats, seed=12345, ode_steps=a.ode_steps
                )
            else:
                session.reset(initial=initial)
            pred = np.stack([session.step(action) for _ in range(a.horizon)])
            gt = fields[2:]
            truths.append(gt)
            predictions.append(pred)
            physics = fluid_physics_metrics(
                gt,
                pred,
                last_context=initial[-1],
                dt=cfg.dt * cfg.save_every,
                substeps=cfg.substeps,
            )
            rows.append(
                {
                    **summarize_physics(physics),
                    "action": action,
                    "name": meta["action_names"][action],
                    "mse": float(np.mean((gt - pred) ** 2)),
                    "vorticity_correlation": float(
                        physics["vorticity_correlation"].mean()
                    ),
                    "mean_velocity_final": pred[-1, ..., :2].mean((0, 1)).tolist(),
                    "simulator_mean_velocity_final": gt[-1, ..., :2]
                    .mean((0, 1))
                    .tolist(),
                }
            )
        for row, pred, truth in zip(rows, predictions, truths):
            row.update(
                steering_response_metrics(
                    pred, truth, predictions[0], truths[0], radius=cfg.action_radius
                )
            )
        responses = [float(np.mean((p - predictions[0]) ** 2)) for p in predictions]
        report = {
            "protocol": {**vars(a), "seed": seed, "out": str(root)},
            "simulator_config": {
                field.name: (
                    -1 if field.name == "held_action_id" else getattr(cfg, field.name)
                )
                for field in dataclass_fields(NS2DConfig)
            },
            "same_initial_context": True,
            "shared_prefix_max_abs_diff": 0.0,
            "simulator_branch_protocol": "shared_state_v1",
            "common_model_noise_seed": 12345,
            "branches": rows,
            "branch_mse_from_noop": responses,
            "simulator_branch_mse_from_noop": [
                float(np.mean((g - truths[0]) ** 2)) for g in truths
            ],
            "interpretation": "Branch separation alone is not correct control; compare simulator response direction and physical metrics.",
        }
        (root / "counterfactual_summary.json").write_text(
            json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n"
        )
        np.savez_compressed(
            root / "counterfactual_arrays.npz",
            gt=np.asarray(truths),
            pred=np.asarray(predictions),
            context=initial,
            simulator_metadata=json.dumps(meta),
        )

    return report, session


def main(a):
    seeds = a.seeds or (a.seed,)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be distinct")
    reports = []
    session = None
    root = Path(a.out)
    for index, seed in enumerate(seeds):
        destination = root if index == 0 else root / f"seed_{seed}"
        report, session = evaluate_seed(a, seed, destination, session)
        reports.append(report)
        print(json.dumps({"completed_seed": seed, "branches": 5}), flush=True)
    combined = dict(reports[0])
    combined.update(
        evaluation_seeds=list(seeds),
        episodes=reports,
        visualized_seed=seeds[0],
        aggregation="Mean of per-episode means; velocity-response cosines exclude no-op, field correlations include all five branches.",
    )
    combined["aggregate_metrics"] = {}
    for key in [
        "velocity_delta_cosine_localized",
        "velocity_delta_cosine",
        "vorticity_correlation",
        "density_correlation",
        "mse",
    ]:
        means = []
        for report in reports:
            rows = (
                report["branches"][1:]
                if key.startswith("velocity_delta")
                else report["branches"]
            )
            values = [r[key] for r in rows if r.get(key) is not None]
            if values:
                means.append(float(np.mean(values)))
        combined["aggregate_metrics"][key] = {
            "mean": float(np.mean(means)) if means else None,
            "episode_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None,
            "episode_means": means,
        }
    (root / "counterfactual_summary.json").write_text(
        json.dumps(jsonable(combined), indent=2, allow_nan=False) + "\n"
    )
    return combined


if __name__ == "__main__":
    main(tyro.cli(Args))
