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
from fluidgenie.training.config_utils import load_toml_config


# ----------------------------
# Entrypoint
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["tokenizer", "rollout"], required=True)
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--stats", type=str, default="", help="Stats .npz for normalization (mean/std)")
    ap.add_argument("--tokenizer_config", type=str, default="", help="Tokenizer TOML config (for codebook/embed/hidden/stats)")
    ap.add_argument("--dynamics_config", type=str, default="", help="Dynamics TOML config (for model/d_model/etc)")

    # tokenizer ckpt
    ap.add_argument("--vq_ckpt", type=str, required=True)

    # tokenizer config (must match training)
    ap.add_argument("--codebook", type=int, default=512)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)

    # tokenizer demo
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--save_gif", type=int, default=0, help="1=save all frames as GIF")
    ap.add_argument("--view", type=str, default="density", choices=["density", "vorticity", "speed", "channel0"])

    # rollout config
    ap.add_argument("--dyn_ckpt", type=str, default="")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--context", type=int, default=2)

    # dynamics model config (must match training)
    ap.add_argument("--model", type=str, default="transformer", choices=["transformer", "maskgit"])
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mask_steps", type=int, default=8)
    ap.add_argument("--kv_cache", type=int, default=1, help="1=use KV cache for transformer rollout")
    ap.add_argument("--rollout_view", type=str, default="density", choices=["density", "vorticity", "speed", "channel0"])

    args = ap.parse_args()

    if args.tokenizer_config:
        cfg = load_toml_config(args.tokenizer_config, section="tokenizer")
        if "codebook" in cfg:
            args.codebook = cfg["codebook"]
        if "embed" in cfg:
            args.embed = cfg["embed"]
        if "hidden" in cfg:
            args.hidden = cfg["hidden"]
        if not args.stats and "stats" in cfg:
            args.stats = cfg["stats"]

    if args.dynamics_config:
        cfg = load_toml_config(args.dynamics_config, section="dynamics")
        if "model" in cfg:
            args.model = cfg["model"]
        if "d_model" in cfg:
            args.d_model = cfg["d_model"]
        if "n_heads" in cfg:
            args.n_heads = cfg["n_heads"]
        if "n_layers" in cfg:
            args.n_layers = cfg["n_layers"]
        if "dropout" in cfg:
            args.dropout = cfg["dropout"]
        if "mask_steps" in cfg:
            args.mask_steps = cfg["mask_steps"]

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
            save_gif=bool(args.save_gif),
            view=args.view,
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
        model_type=args.model,
        use_kv_cache=bool(args.kv_cache),
        mask_steps=args.mask_steps,
        view=args.rollout_view,
        stats_path=args.stats if args.stats else None,
    )


if __name__ == "__main__":
    main()
