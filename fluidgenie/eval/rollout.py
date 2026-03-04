from __future__ import annotations

import tyro

from configs.eval_configs import RolloutConfig, apply_ckpt_config_to_rollout
from fluidgenie.eval.rollout_runner import run_rollout as _run_rollout


def main() -> None:
    cfg = tyro.cli(RolloutConfig)
    cfg = apply_ckpt_config_to_rollout(cfg)
    _run_rollout(cfg)


if __name__ == "__main__":
    main()
