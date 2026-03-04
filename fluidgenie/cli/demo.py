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
from configs.eval_configs import DemoArgs, rollout_config_from_demo


# ----------------------------
# Entrypoint
# ----------------------------

def main() -> None:
    args = tyro.cli(DemoArgs)

    if args.mode == "tokenizer":
        if not args.tokenizer.vq_ckpt:
            raise ValueError("--vq_ckpt is required for --mode tokenizer")
        save_tokenizer_recon(
            npz_path=args.npz,
            vq_ckpt=args.tokenizer.vq_ckpt,
            out_dir=args.out,
            frame=args.frame,
            codebook_size=args.tokenizer.codebook,
            embed_dim=args.tokenizer.embed,
            hidden=args.tokenizer.hidden,
            stats_path=args.tokenizer.stats if args.tokenizer.stats else None,
            save_gif=args.save_gif,
            view=args.view,
            tokenizer_arch=args.tokenizer.arch,
            patch_size=args.tokenizer.patch_size,
            model_dim=args.tokenizer.model_dim,
            num_blocks=args.tokenizer.num_blocks,
            num_heads=args.tokenizer.num_heads,
            dropout=args.tokenizer.dropout,
            codebook_dropout=args.tokenizer.codebook_dropout,
            bg_thresh=args.tokenizer.bg_thresh,
        )
        return

    if args.mode != "rollout":
        raise ValueError("--mode must be 'tokenizer' or 'rollout'")
    if not args.rollout.dyn_ckpt:
        raise ValueError("--dyn_ckpt is required for --mode rollout")

    run_rollout(rollout_config_from_demo(args))


if __name__ == "__main__":
    main()
