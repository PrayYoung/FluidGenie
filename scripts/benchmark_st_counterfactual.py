"""Evaluate calibrated LAM steering only after its discovery gate passes."""

from dataclasses import dataclass
from pathlib import Path
import json
import tempfile
import numpy as np
import tyro
from configs.eval_configs import RolloutConfig, apply_ckpt_config_to_rollout
from fluidgenie.eval.rollout_runner import run_rollout
from fluidgenie.eval.metrics import (
    fluid_physics_metrics,
    jsonable,
    steering_response_metrics,
    summarize_physics,
)


@dataclass
class Args:
    tokenizer: str
    dynamics: str
    lam: str
    stats: str
    discovery: str
    simulator_branches: str
    out: str


def main(a):
    discovery = json.loads(Path(a.discovery).read_text())
    if not discovery["discovery_gate_passed"] or not discovery.get(
        "five_action_mapping_gate", False
    ):
        raise ValueError(
            "Five-action semantic LAM steering requires passing discovery and mapping gates"
        )
    from fluidgenie.eval.action_discovery import require_discovery_certificate

    require_discovery_certificate(discovery)
    arrays = np.load(a.simulator_branches)
    truths = arrays["gt"]
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = []
    metadata = (
        json.loads(str(arrays["simulator_metadata"]))
        if "simulator_metadata" in arrays
        else {}
    )
    with tempfile.TemporaryDirectory() as temp:
        for action in range(5):
            # The counterfactual archive carries the shared pre-intervention context.
            fields = np.concatenate([arrays["context"], truths[action]], axis=0)
            path = Path(temp) / "episode.npy"
            np.save(path, fields)
            cfg = RolloutConfig(
                str(path),
                a.tokenizer,
                a.dynamics,
                str(root / f"action_{action}"),
                context=2,
                horizon=len(truths[action]),
                stats_path=a.stats,
                tokenizer_arch="st",
                model_type="st_maskgit",
                use_lam=True,
                lam_ckpt=a.lam,
                held_action_id=discovery["calibrated_code_by_action"][action],
                seed=12345,
                skip_split_check=True,
            )
            run_rollout(apply_ckpt_config_to_rollout(cfg))
            result = np.load(root / f"action_{action}/rollout_arrays.npz")
            predictions.append(result["pred"])
            physics = fluid_physics_metrics(
                result["gt"],
                result["pred"],
                last_context=arrays["context"][-1],
                dt=metadata.get("dt", 1.0) * metadata.get("save_every", 1),
                substeps=metadata.get("substeps", 1),
            )
            rows.append(
                {
                    **summarize_physics(physics),
                    "action": action,
                    "latent_code": cfg.held_action_id,
                    "mse": float(np.mean((result["gt"] - result["pred"]) ** 2)),
                    "vorticity_correlation": float(
                        physics["vorticity_correlation"].mean()
                    ),
                }
            )
    for row, pred, truth in zip(rows, predictions, truths):
        row.update(
            steering_response_metrics(
                pred,
                truth,
                predictions[0],
                truths[0],
                radius=metadata.get("action_radius", 0.14),
            )
        )
    (root / "st_counterfactual_summary.json").write_text(
        json.dumps(
            jsonable(
                {
                    "branches": rows,
                    "same_initial_context": True,
                    "calibration_source": "training labels only",
                    "discovery_passed": True,
                }
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
