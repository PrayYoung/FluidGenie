FluidGenie
==========

Token-based fluid dynamics modeling with two pipelines:

1) **Conv VQ + AR/MaskGIT dynamics** (original pipeline)
2) **ST-VQ + ST-MaskGIT + optional LAM** (Jafar-style pipeline)

This repo uses `tyro` for CLI configs and `uv` for running.

Setup
-----
```bash
uv sync
```

Data Generation (PhiFlow)
-------------------------
```bash
uv run python -m fluidgenie.data.gen_phiflow_ns2d \
  --out data/ns2d \
  --episodes 200 \
  --steps 200 \
  --res 128 \
  --density 1
```

Optional normalization stats:
```bash
uv run python -m fluidgenie.data.compute_stats \
  --data data/ns2d \
  --out data/ns2d_stats.npz
```

Tokenizer Training
------------------

### A) Conv VQ (original)
```bash
uv run python -m fluidgenie.training.train_tokenizer \
  --data data/ns2d \
  --out runs/vq \
  --stats data/ns2d_stats.npz
```

### B) ST VQ (Jafar-style)
```bash
uv run python -m fluidgenie.training.train_tokenizer_st \
  --data data/ns2d \
  --out runs/vq_st \
  --seq-len 4 \
  --stats data/ns2d_stats.npz
```

LAM Training (optional)
-----------------------
```bash
uv run python -m fluidgenie.training.train_lam \
  --data data/ns2d \
  --out runs/lam \
  --seq-len 8 \
  --stats data/ns2d_stats.npz
```

Dynamics Training
-----------------

### A) AR / MaskGIT (original pipeline)
```bash
uv run python -m fluidgenie.training.train_dynamics \
  --data data/ns2d \
  --vq-ckpt runs/vq/latest.ckpt \
  --out runs/dyn \
  --model transformer
```

MaskGIT variant:
```bash
uv run python -m fluidgenie.training.train_dynamics \
  --data data/ns2d \
  --vq-ckpt runs/vq/latest.ckpt \
  --out runs/dyn_maskgit \
  --model maskgit
```

### B) ST-MaskGIT (Jafar-style)
```bash
uv run python -m fluidgenie.training.train_dynamics_st \
  --data data/ns2d \
  --vq-ckpt runs/vq_st/latest.ckpt \
  --out runs/dyn_st \
  --model st_maskgit
```

With LAM conditioning:
```bash
uv run python -m fluidgenie.training.train_dynamics_st \
  --data data/ns2d \
  --vq-ckpt runs/vq_st/latest.ckpt \
  --out runs/dyn_st_lam \
  --model st_maskgit \
  --use-lam True \
  --lam-ckpt runs/lam/latest.ckpt
```

Evaluation / Demo
-----------------

Tokenizer recon:
```bash
uv run python -m fluidgenie.cli.demo \
  --mode tokenizer \
  --npz data/ns2d/episode_000000.npz \
  --vq-ckpt runs/vq/latest.ckpt \
  --out demo/tokenizer \
  --view density
```

Rollout (AR):
```bash
uv run python -m fluidgenie.cli.demo \
  --mode rollout \
  --npz data/ns2d/episode_000000.npz \
  --vq-ckpt runs/vq/latest.ckpt \
  --dyn-ckpt runs/dyn/latest.ckpt \
  --out demo/rollout \
  --model transformer
```

Rollout (ST-MaskGIT + ST tokenizer):
```bash
uv run python -m fluidgenie.cli.demo \
  --mode rollout \
  --npz data/ns2d/episode_000000.npz \
  --vq-ckpt runs/vq_st/latest.ckpt \
  --dyn-ckpt runs/dyn_st/latest.ckpt \
  --out demo/rollout_st \
  --model st_maskgit \
  --tokenizer-arch st
```

Codebook usage (collapse check):
```bash
uv run python -m fluidgenie.cli.eval_codebook \
  --data data/ns2d \
  --vq-ckpt runs/vq/latest.ckpt
```

Notes
-----
- `tokenizer-arch` can be `conv` or `st`.
- For ST-MaskGIT, the dynamics model runs spatial-temporal attention on patch tokens.
- LAM conditioning is optional and only used when `use-lam=True` and `lam-ckpt` is provided.
