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

import argparse
from fluidgenie.eval.viz import save_tokenizer_recon
from fluidgenie.eval.rollout import run_rollout


# ----------------------------
# Entrypoint
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["tokenizer", "rollout"], required=True)
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--stats", type=str, default="", help="Stats .npz for normalization (mean/std)")

    # tokenizer ckpt
    ap.add_argument("--vq_ckpt", type=str, required=True)

    # tokenizer config (must match training)
    ap.add_argument("--codebook", type=int, default=512)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)

    # tokenizer demo
    ap.add_argument("--frame", type=int, default=0)

    # rollout config
    ap.add_argument("--dyn_ckpt", type=str, default="")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--context", type=int, default=2)

    # dynamics model config (must match training)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)

    args = ap.parse_args()

    if args.mode == "tokenizer":
        save_tokenizer_recon(
            npz_path=args.npz,
            vq_ckpt=args.vq_ckpt,
            out_dir=args.out,
            frame=args.frame,
            codebook_size=args.codebook,
            embed_dim=args.embed,
            hidden=args.hidden,
            stats_path=args.stats if args.stats else None,
        )
        return

    # rollout mode
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
        stats_path=args.stats if args.stats else None,
    )


if __name__ == "__main__":
    main()
