"""
CLI: generate fluid dataset with PhiFlow NS2D generator (NPZ episodes).

This CLI matches the current NS2DConfig fields in:
  fluidgenie/data/gen_phiflow_ns2d.py

Example:
  uv run python -m fluidgenie.cli.gen_data \
    --out data/ns2d \
    --episodes 200 \
    --steps 200 \
    --res 128 \
    --dt 0.1 \
    --viscosity 0.001 \
    --substeps 1 \
    --density 1 \
    --implicit 0 \
    --prefix episode
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fluidgenie.data.gen_phiflow_ns2d import NS2DConfig, generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output directory for npz episodes")
    ap.add_argument("--episodes", type=int, default=10, help="Number of episodes to generate")
    ap.add_argument("--prefix", type=str, default="episode", help="Filename prefix")

    # NS2DConfig fields
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--viscosity", type=float, default=0.001)
    ap.add_argument("--implicit", type=int, default=0, help="1=use diffuse.implicit, 0=explicit")
    ap.add_argument("--density", type=int, default=1, help="1=with density channel, 0=velocity-only")
    ap.add_argument("--substeps", type=int, default=1, help="internal substeps per saved frame")
    ap.add_argument("--save-every", type=int, default=1, help="save one frame every N steps")
    ap.add_argument("--noise", type=float, default=0.10, help="initial velocity noise scale")

    # Future forcing (already present in config)
    ap.add_argument("--forcing_strength", type=float, default=0.0)
    ap.add_argument("--forcing_radius", type=float, default=0.08)
    ap.add_argument("--density_radius", type=float, default=0.08)

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = NS2DConfig(
        seed=args.seed,
        steps=args.steps,
        dt=args.dt,
        resolution=args.res,
        viscosity=args.viscosity,
        use_implicit_diffusion=bool(args.implicit),
        with_density=bool(args.density),
        substeps=args.substeps,
        save_every=args.save_every,
        init_velocity_noise=args.noise,
        forcing_strength=args.forcing_strength,
        forcing_radius=args.forcing_radius,
        density_radius=args.density_radius,
    )

    generate_dataset(
        out_dir=str(out_dir),
        episodes=args.episodes,
        cfg=cfg,
        name_prefix=args.prefix,
    )

    print(f"Done. Wrote {args.episodes} episodes to {out_dir}")


if __name__ == "__main__":
    main()
