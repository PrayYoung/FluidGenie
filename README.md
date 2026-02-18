🌊 **FluidGenie**
============

[![JAX](https://img.shields.io/badge/JAX-0.4.20+-blue.svg)](https://github.com/google/jax)
[![FLAX](https://img.shields.io/badge/Flax-0.10.7+-blue.svg)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**FluidGenie** is a high performance, token-based generative framework for modeling complex spatiotemporal physical systems (fluid dynamics) using **JAX/Flax**, inspired by Google's **Genie** (https://arxiv.org/abs/2402.15391).

By unifying Vision Transformer (ViT) and Large Language Model (LLM) paradigms, FluidGenie compresses continuous fluid simulation into discrete token grids and models their physical evolution using advanced autoregressive and non-autoregressive (MaskGIT) sequence modeling.

## ⭐ Key Highlights
- Dual state‑of‑the‑art pipelines:
  - Standard pipeline (Conv VQ + AR/MaskGIT)
  - ST pipeline (ST‑VQ + ST‑MaskGIT, optional LAM)
- Hardware‑accelerated engineering on JAX/Flax (GPU/TPU‑friendly)
- Physical fidelity validation via vorticity/gradient losses and visual diagnostics
- End‑to‑end simulation pipeline: data → tokenize → dynamics → rollout

---

## What You Get
- **Conv VQ + AR/MaskGIT** (stable baseline)
- **ST‑VQ + ST‑MaskGIT + optional LAM** (Jafar‑style)
- **PhiFlow** NS2D data generator
- **Rollout + visualization** with GIFs
- **Orbax checkpoints** (directory format, still compatible with old `.ckpt` files)

---

## Quick Start (Minimal)
```bash
uv sync

# 1) Generate data
uv run python -m fluidgenie.data.gen_phiflow_ns2d \
  --out data/raw/ns2d \
  --episodes 50 \
  --steps 120

# 2) Train tokenizer (Conv VQ)
uv run python -m fluidgenie.training.train_tokenizer_base \
  --data data/raw/ns2d \
  --out runs/tokenizer/base

# 3) Train dynamics (AR)
uv run python -m fluidgenie.training.train_dynamics_base \
  --data data/raw/ns2d \
  --vq-ckpt runs/tokenizer/base/latest \
  --out runs/dynamics/base \
  --model transformer

# 4) Rollout
uv run python -m fluidgenie.cli.demo \
  --mode rollout \
  --npz data/raw/ns2d/episode_000000.npy \
  --vq-ckpt runs/tokenizer/base/latest \
  --dyn-ckpt runs/dynamics/base/latest \
  --out demo/rollout/base
```

---

## Pipeline A: Conv VQ + AR/MaskGIT

### Tokenizer
```bash
uv run python -m fluidgenie.training.train_tokenizer_base \
  --data data/raw/ns2d \
  --out runs/tokenizer/base \
  --stats data/stats/ns2d_stats.npz
```

### Dynamics (AR)
```bash
uv run python -m fluidgenie.training.train_dynamics_base \
  --data data/raw/ns2d \
  --vq-ckpt runs/tokenizer/base/latest \
  --out runs/dynamics/base \
  --model transformer
```

### Dynamics (MaskGIT)
```bash
uv run python -m fluidgenie.training.train_dynamics_base \
  --data data/raw/ns2d \
  --vq-ckpt runs/tokenizer/base/latest \
  --out runs/dynamics/base_maskgit \
  --model maskgit
```

---

## Pipeline B: ST‑VQ + ST‑MaskGIT (+ LAM)

### ST Tokenizer
```bash
uv run python -m fluidgenie.training.train_tokenizer_st \
  --data data/raw/ns2d \
  --out runs/tokenizer/st \
  --seq-len 4
```

### ST Dynamics
```bash
uv run python -m fluidgenie.training.train_dynamics_st \
  --data data/raw/ns2d \
  --vq-ckpt runs/tokenizer/st/latest \
  --out runs/dynamics/st \
  --model st_maskgit \
```

### Optional LAM
```bash
uv run python -m fluidgenie.training.train_lam \
  --data data/raw/ns2d \
  --out runs/lam \
```

### ST Dynamics + LAM
```bash
uv run python -m fluidgenie.training.train_dynamics_st \
  --data data/raw/ns2d \
  --vq-ckpt runs/tokenizer/st/latest \
  --out runs/dynamics/st_lam \
  --model st_maskgit \
  --use-lam \
  --lam-ckpt runs/lam/latest \
```

---

## Visualization & Eval

### Demo A: Base Tokenizer + Base Dynamics
```bash
uv run python -m fluidgenie.cli.demo \
  --mode tokenizer \
  --npz data/raw/ns2d/episode_000000.npy \
  --vq-ckpt runs/tokenizer/base/latest \
  --out demo/tokenizer/base \
  --view density
```

```bash
uv run python -m fluidgenie.cli.demo \
  --mode rollout \
  --npz data/raw/ns2d/episode_000000.npy \
  --vq-ckpt runs/tokenizer/base/latest \
  --dyn-ckpt runs/dynamics/base/latest \
  --out demo/rollout/base
```

### Demo B: ST Tokenizer + ST‑MaskGIT
```bash
uv run python -m fluidgenie.cli.demo \
  --mode tokenizer \
  --npz data/raw/ns2d/episode_000000.npy \
  --vq-ckpt runs/tokenizer/st/latest \
  --out demo/tokenizer/st \
  --tokenizer-arch st
```

```bash
uv run python -m fluidgenie.cli.demo \
  --mode rollout \
  --npz data/raw/ns2d/episode_000000.npy \
  --vq-ckpt runs/tokenizer/st/latest \
  --dyn-ckpt runs/dynamics/st/latest \
  --out demo/rollout/st \
  --model st_maskgit \
  --tokenizer-arch st
```

### Codebook Usage (collapse check)
```bash
uv run python -m fluidgenie.cli.eval_codebook \
  --data data/raw/ns2d \
  --vq-ckpt runs/tokenizer/base/latest
```

---

## Optional: Normalization Stats
```bash
uv run python -m fluidgenie.data.compute_stats \
  --data data/raw/ns2d \
  --out data/stats/ns2d_stats.npz
```

---

## Perf Tips
- Use Grain workers to parallelize data loading:
  - `--grain-workers 4`
- For large rollouts, prefer `--model st_maskgit` to avoid slow AR decode.

---

## Notes
- `tokenizer-arch` can be `conv` or `st`.
- Checkpoints are saved as **directories** with Orbax (e.g. `runs/tokenizer/base/latest`).
- Old `.ckpt` files are still loadable.
