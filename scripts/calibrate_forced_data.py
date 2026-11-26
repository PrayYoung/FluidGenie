"""One predeclared forced-flow candidate, evaluated without adaptive retuning."""

from dataclasses import asdict, replace
from pathlib import Path
import json
import tyro
from dataclasses import dataclass
from configs.gen_data_configs import forced_control_config
from fluidgenie.data.gen_phiflow_ns2d import generate_dataset
from fluidgenie.eval.metrics import jsonable
from fluidgenie.data.quality import validate


@dataclass
class Args:
    out: str


def candidate_config():
    return forced_control_config(steps=160, burn_in=1000)


def main(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    plan = dict(
        candidate=asdict(candidate_config()),
        resolutions=[32, 64],
        seeds=[99991, 99992],
        mandatory_gates=["finite_gate", "persistent_motion_gate", "stationarity_gate"],
        motion_floor=1e-3,
        preferred_mean_motion=1e-2,
        adaptive_retuning=False,
        interpretation="Finite-window calibration only; passing does not prove turbulence or indefinite stationarity.",
    )
    plan_path = root / "plan.json"
    if plan_path.exists():
        raise FileExistsError(
            "Use a fresh calibration output; do not overwrite outcomes"
        )
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for resolution in plan["resolutions"]:
        for seed in plan["seeds"]:
            data = root / f"r{resolution}_seed{seed}"
            generate_dataset(
                str(data),
                1,
                replace(candidate_config(), resolution=resolution, seed=seed),
            )
            result = validate(data)
            results.append(dict(resolution=resolution, seed=seed, **result))
            (root / "results.json").write_text(
                json.dumps(
                    jsonable(dict(plan=plan, results=results)),
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
    passed = all(all(r[k] for k in plan["mandatory_gates"]) for r in results)
    report = dict(plan=plan, results=results, all_mandatory_gates_passed=passed)
    (root / "results.json").write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    main(tyro.cli(Args))
