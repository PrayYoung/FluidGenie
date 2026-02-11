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

import tyro

from fluidgenie.eval.viz import save_tokenizer_recon
from fluidgenie.eval.rollout import run_rollout
from configs.eval_configs import DemoArgs


# ----------------------------
# Entrypoint
# ----------------------------

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
            tokenizer_arch=args.tokenizer_arch,
            patch_size=args.patch_size,
            model_dim=args.model_dim,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout=args.dropout,
            codebook_dropout=args.codebook_dropout,
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
        tokenizer_arch=args.tokenizer_arch,
        patch_size=args.patch_size,
        model_dim=args.model_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        tokenizer_dropout=args.dropout,
        codebook_dropout=args.codebook_dropout,
        use_lam=args.use_lam,
        lam_ckpt=args.lam_ckpt,
        lam_model_dim=args.lam_model_dim,
        lam_latent_dim=args.lam_latent_dim,
        lam_num_latents=args.lam_num_latents,
        lam_patch_size=args.lam_patch_size,
        lam_num_blocks=args.lam_num_blocks,
        lam_num_heads=args.lam_num_heads,
        lam_dropout=args.lam_dropout,
        lam_codebook_dropout=args.lam_codebook_dropout,
    )


if __name__ == "__main__":
    main()
