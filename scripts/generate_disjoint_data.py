"""Generate periodic forced train/held-out data with training-only z-score stats."""

from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np
import tyro
from dataclasses import dataclass
from configs.gen_data_configs import NS2DConfig, forced_control_config
from fluidgenie.data.gen_phiflow_ns2d import generate_dataset
from fluidgenie.data.compute_stats import compute_stats
from fluidgenie.data.splits import assert_disjoint


@dataclass
class Args:
    out: str = "data/forced_periodic_v2"
    train_episodes: int = 30
    heldout_episodes: int = 10
    train_seed: int = 0
    heldout_seed: int = 200003
    resolution: int = 64
    steps: int = 160
    burn_in: int = 1000
    action_mode: str = "none"
    forcing_strength: float | None = None
    forced_preset: str = "unsteady_v4"


def main(args):
    train_seeds = set(range(args.train_seed, args.train_seed + args.train_episodes))
    if train_seeds & set(
        range(args.heldout_seed, args.heldout_seed + args.heldout_episodes)
    ):
        raise ValueError("Requested seed ranges overlap")
    root = Path(args.out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "Choose an empty output directory to avoid mixing protocols"
        )
    cfg = NS2DConfig(
        seed=args.train_seed,
        resolution=args.resolution,
        steps=args.steps,
        burn_in=args.burn_in,
        action_mode=args.action_mode,
        forcing_strength=0.02
        if args.forcing_strength is None
        else args.forcing_strength,
    )
    if args.forced_preset not in {"legacy_v3", "unsteady_v4"}:
        raise ValueError("Unknown forced-control preset")
    if args.action_mode == "none" and args.forced_preset == "unsteady_v4":
        cfg = forced_control_config(
            seed=args.train_seed,
            resolution=args.resolution,
            steps=args.steps,
            burn_in=args.burn_in,
        )
        if args.forcing_strength is not None:
            cfg = replace(cfg, forcing_strength=args.forcing_strength)
    if args.action_mode == "joystick5_localized":
        cfg = replace(
            cfg,
            action_strength=5.0,
            action_hold_min=2,
            action_hold_max=4,
            density_radius=0.07,
            density_decay=0.8,
            density_source_strength=2.5,
            viscosity=0.002,
            substeps=8,
            forcing_strength=0.25
            if args.forcing_strength is None
            else args.forcing_strength,
        )
    generate_dataset(str(root / "train"), args.train_episodes, cfg)
    generate_dataset(
        str(root / "heldout"),
        args.heldout_episodes,
        replace(cfg, seed=args.heldout_seed),
    )
    manifest = assert_disjoint(root / "train", root / "heldout")
    mean, std, _, _ = compute_stats(str(root / "train"))
    np.savez(root / "train_stats.npz", mean=mean, std=std)
    manifest.update(
        protocol=args.forced_preset,
        config=asdict(args),
        cli_args=asdict(args),
        generator_config=asdict(cfg),
        statistics_source="train",
    )
    (root / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
