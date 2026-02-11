"""
Unified demo entry for FluidGenie.

Modes:
  1) tokenizer: VQ recon + token map (good for README)
  2) rollout:   token-space autoregressive rollout -> decode -> GIF

Examples:

# 1) Tokenizer recon (requires VQ ckpt)
uv run python -m fluidgenie.cli.demo \
  --mode tokenizer \
  --npz data/ns2d/episode_000000.npz \
  --vq_ckpt runs/vq/latest.ckpt \
  --out demo/tokenizer \
  --frame 0 \
  --codebook 512 --embed 64 --hidden 128

# 2) Rollout GIF (requires VQ ckpt + Dynamics ckpt)
uv run python -m fluidgenie.cli.demo \
  --mode rollout \
  --npz data/ns2d/episode_000000.npz \
  --vq_ckpt runs/vq/latest.ckpt \
  --dyn_ckpt runs/dyn/latest.ckpt \
  --out demo/rollout \
  --start 0 --horizon 60 --context 2 \
  --codebook 512 --embed 64 --hidden 128 \
  --d_model 256 --n_heads 8 --n_layers 6 --dropout 0.1
"""

from __future__ import annotations

from dataclasses import dataclass

import tyro

from fluidgenie.eval.viz import save_tokenizer_recon
from fluidgenie.eval.rollout import run_rollout


# ----------------------------
# Entrypoint
# ----------------------------

@dataclass
class DemoArgs:
    mode: str  # "tokenizer" | "rollout"
    npz: str
    out: str
    stats: str = ""

    vq_ckpt: str = ""
    codebook: int = 512
    embed: int = 64
    hidden: int = 128

    frame: int = 0
    save_gif: bool = False
    view: str = "density"  # density | vorticity | speed | channel0

    dyn_ckpt: str = ""
    start: int = 0
    horizon: int = 60
    context: int = 2

    model: str = "transformer"  # transformer | maskgit
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    mask_steps: int = 8
    kv_cache: bool = True
    rollout_view: str = "density"


def main():
    args = tyro.cli(DemoArgs)

    if args.mode == "tokenizer":
        if not args.vq_ckpt:
            raise ValueError("--vq_ckpt is required for --mode tokenizer")
        save_tokenizer_recon(
            npz_path=args.npz,
            vq_ckpt=args.vq_ckpt,
            out_dir=args.out,
            frame=args.frame,
            codebook_size=args.codebook,
            embed_dim=args.embed,
            hidden=args.hidden,
            stats_path=args.stats if args.stats else None,
            save_gif=args.save_gif,
            view=args.view,
        )
        return

    if args.mode != "rollout":
        raise ValueError("--mode must be 'tokenizer' or 'rollout'")
    if not args.dyn_ckpt:
        raise ValueError("--dyn_ckpt is required for --mode rollout")

    run_rollout(
        npz_path=args.npz,
        vq_ckpt=args.vq_ckpt,
        dyn_ckpt=args.dyn_ckpt,
        out_dir=args.out,
        start=args.start,
        horizon=args.horizon,
        context=args.context,
        codebook_size=args.codebook,
        embed_dim=args.embed,
        hidden=args.hidden,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        model_type=args.model,
        use_kv_cache=args.kv_cache,
        mask_steps=args.mask_steps,
        view=args.rollout_view,
        stats_path=args.stats if args.stats else None,
    )


if __name__ == "__main__":
    main()
