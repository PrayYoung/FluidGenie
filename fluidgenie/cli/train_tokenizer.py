"""
CLI wrapper for tokenizer training.

Example:
  uv run python -m fluidgenie.cli.train_tokenizer \
    --data data/ns2d \
    --out runs/vq \
    --steps 20000 \
    --batch 8 \
    --codebook 1024
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--codebook", type=int, default=1024)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cmd = [
        sys.executable, "-m", "fluidgenie.training.train_tokenizer",
        "--data", args.data,
        "--out", args.out,
        "--steps", str(args.steps),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--codebook", str(args.codebook),
        "--embed", str(args.embed),
        "--hidden", str(args.hidden),
        "--seed", str(args.seed),
    ]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
