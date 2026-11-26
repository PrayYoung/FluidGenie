"""Report finite states, persistent motion, and a predeclared stationarity gate."""

from dataclasses import dataclass
from pathlib import Path
import json
import tyro
from fluidgenie.data.quality import validate


@dataclass
class Args:
    data: str
    out: str
    require_stationarity: bool = True



def main(args):
    result = validate(args.data)
    result["require_stationarity"] = args.require_stationarity
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    from fluidgenie.eval.metrics import jsonable

    path.write_text(json.dumps(jsonable(result), indent=2, allow_nan=False) + "\n")
    required = ["finite_gate", "persistent_motion_gate"]
    if args.require_stationarity:
        required.append("stationarity_gate")
    if not all(result[k] for k in required):
        raise SystemExit(
            "Required data gate failed: "
            + ", ".join(k for k in required if not result[k])
        )
    return result


if __name__ == "__main__":
    main(tyro.cli(Args))
