"""
CLI: generate fluid dataset with PhiFlow NS2D generator (NPY episodes).

This CLI matches the current NS2DConfig fields in:
  fluidgenie/data/gen_phiflow_ns2d.py

Example:
  uv run python -m fluidgenie.cli.gen_data \
    --out data/raw/ns2d \
    --episodes 200 \
    --steps 200 \
    --res 128 \
    --dt 0.1 \
    --viscosity 0.001 \
    --substeps 1 \
    --density 1 \
    --prefix episode
"""

from __future__ import annotations

from pathlib import Path

import tyro

from configs.gen_data_configs import GenDataArgs
from fluidgenie.data.gen_phiflow_ns2d import generate_dataset, config_from_args


def main() -> None:
    args = tyro.cli(GenDataArgs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = config_from_args(args)

    generate_dataset(
        out_dir=str(out_dir),
        episodes=args.episodes,
        cfg=cfg,
        name_prefix=args.prefix,
    )

    print(f"Done. Wrote {args.episodes} episodes to {out_dir}")


if __name__ == "__main__":
    main()
