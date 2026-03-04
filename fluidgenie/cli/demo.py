"""
Unified demo entry for FluidGenie.

Modes:
  1) tokenizer: VQ recon + token map (good for README)
  2) rollout:   token-space autoregressive rollout -> decode -> GIF

Examples:

# 1) Tokenizer recon (requires VQ ckpt)
uv run python -m fluidgenie.cli.demo \
  --mode tokenizer \
  --npz data/raw/ns2d/episode_000000.npy \
  --vq_ckpt runs/tokenizer/base/latest \
  --out demo/tokenizer/base \
  --frame 0 \
  --codebook 512 --embed 64 --hidden 128

# 2) Rollout GIF (requires VQ ckpt + Dynamics ckpt)
uv run python -m fluidgenie.cli.demo \
  --mode rollout \
  --npz data/raw/ns2d/episode_000000.npy \
  --vq_ckpt runs/tokenizer/base/latest \
  --dyn_ckpt runs/dynamics/base/latest \
  --out demo/rollout/base \
  --start 0 --horizon 60 --context 2 \
  --codebook 512 --embed 64 --hidden 128 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1
"""

from __future__ import annotations

import tyro

from fluidgenie.eval.viz import save_tokenizer_recon
from fluidgenie.eval.rollout_runner import run_rollout
from configs.eval_configs import (
    DemoArgs,
    rollout_config_from_demo,
    tokenizer_recon_kwargs_from_demo,
    apply_ckpt_config_to_demo,
)


# ----------------------------
# Entrypoint
# ----------------------------

def main() -> None:
    args = tyro.cli(DemoArgs)
    args = apply_ckpt_config_to_demo(args)

    if args.mode == "tokenizer":
        if not args.tokenizer.vq_ckpt:
            raise ValueError("--vq_ckpt is required for --mode tokenizer")
        save_tokenizer_recon(**tokenizer_recon_kwargs_from_demo(args))
        return

    if args.mode != "rollout":
        raise ValueError("--mode must be 'tokenizer' or 'rollout'")
    if not args.rollout.dyn_ckpt:
        raise ValueError("--dyn_ckpt is required for --mode rollout")

    run_rollout(rollout_config_from_demo(args))


if __name__ == "__main__":
    main()
