"""
CLI wrapper for dynamics training.

Example:
  uv run python -m fluidgenie.cli.train_dynamics \
    --data data/ns2d \
    --vq_ckpt runs/vq/latest.ckpt \
    --out runs/dyn \
    --steps 20000
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--vq_ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--codebook", type=int, default=512)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--tb", type=int, default=1)
    ap.add_argument("--stats", type=str, default="")

    args = ap.parse_args()

    cmd = [
        sys.executable, "-m", "fluidgenie.training.train_dynamics",
        "--data", args.data,
        "--vq_ckpt", args.vq_ckpt,
        "--out", args.out,
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--context", str(args.context),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--codebook", str(args.codebook),
        "--embed", str(args.embed),
        "--hidden", str(args.hidden),
        "--d_model", str(args.d_model),
        "--n_heads", str(args.n_heads),
        "--n_layers", str(args.n_layers),
        "--dropout", str(args.dropout),
        "--log_every", str(args.log_every),
        "--tb", str(args.tb),
        "--stats", str(args.stats),
    ]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
