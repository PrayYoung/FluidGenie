from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Shared defaults to keep tokenizer/dynamics consistent
DEFAULT_CODEBOOK = 512
DEFAULT_EMBED = 64
DEFAULT_HIDDEN = 128
DEFAULT_PATCH_SIZE = 4
DEFAULT_BG_THRESH = 0.0


@dataclass
class TokenizerConfig:
    data: str
    out: str
    architecture_version: int = 2
    batch: int = 128
    steps: int = 10000
    lr: float = 3e-4
    codebook: int = DEFAULT_CODEBOOK
    embed: int = DEFAULT_EMBED
    hidden: int = DEFAULT_HIDDEN
    seed: int = 0
    checkpoint_every: int = 1000
    log_every: int = 100
    tb: int = 1
    stats: str = ""
    loss_alpha: float = 1.0
    loss_beta: float = 0.25
    loss_gamma: float = 0.1

    # Optional spatial-temporal tokenizer
    arch: str = "conv"  # conv | st
    patch_size: int = DEFAULT_PATCH_SIZE
    model_dim: int = 256
    num_blocks: int = 6
    num_heads: int = 8
    dropout: float = 0.0
    codebook_dropout: float = 0.0
    bg_thresh: float = DEFAULT_BG_THRESH
    seq_len: int = 2
    restart_every: int = 100
    restart_until_fraction: float = 0.8
    grain_workers: int = 8


@dataclass
class DynamicsConfig:
    data: str
    vq_ckpt: str
    out: str
    architecture_version: int = 2
    steps: int = 20000
    batch: int = 4
    context: int = 2
    pred: int = 1
    frame_stride: int = 1
    lr: float = 3e-4
    seed: int = 0

    codebook: int = DEFAULT_CODEBOOK
    embed: int = DEFAULT_EMBED
    hidden: int = DEFAULT_HIDDEN

    model: str = "transformer"
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    checkpoint_every: int = 1000
    log_every: int = 50
    tb: int = 1
    stats: str = ""

    # MaskGIT options (only used when model = "maskgit")
    mask_ratio_min: float = 0.1
    mask_ratio_max: float = 1.0
    mask_full_prob: float = 0.2
    mask_schedule: str = "cosine"
    mask_steps: int = 8
    bos_token_id: int = 0
    grain_workers: int = 8

    # Optional LAM conditioning for spatial-temporal dynamics
    use_lam: bool = False
    lam_ckpt: str = ""
    lam_model_dim: int = 256
    lam_latent_dim: int = 64
    lam_num_latents: int = 128
    lam_patch_size: int = 4
    lam_num_blocks: int = 6
    lam_num_heads: int = 8
    lam_dropout: float = 0.0
    lam_codebook_dropout: float = 0.0

    # ST tokenizer settings (for train_dynamics_st)
    tok_patch_size: int = DEFAULT_PATCH_SIZE
    tok_model_dim: int = 256
    tok_num_blocks: int = 6
    tok_num_heads: int = 8
    tok_dropout: float = 0.0
    tok_codebook_dropout: float = 0.0
    bg_thresh: float = DEFAULT_BG_THRESH


@dataclass
class LAMConfig:
    data: str
    out: str
    architecture_version: int = 4
    resume: bool = False
    steps: int = 20000
    batch: int = 4
    seq_len: int = 8
    lr: float = 3e-4
    seed: int = 0
    checkpoint_every: int = 1000
    log_every: int = 50
    tb: int = 1
    stats: str = ""

    restart_every: int = 100
    restart_until_fraction: float = 0.8
    vq_beta: float = 0.25
    model_dim: int = 256
    latent_dim: int = 64
    num_latents: int = 128
    patch_size: int = 4
    num_blocks: int = 6
    num_heads: int = 8
    dropout: float = 0.0
    codebook_dropout: float = 0.0
    grain_workers: int = 8


@dataclass
class FlowMatchingConfig:
    """Configuration for the isolated direct state-space flow-matching baseline."""

    data: str
    out: str
    resume: bool = False
    steps: int = 20000
    batch: int = 32
    context: int = 2
    lr: float = 3e-4
    seed: int = 0
    stats: str = ""
    checkpoint_every: int = 1000
    log_every: int = 50
    tb: int = 1
    grain_workers: int = 8

    # Plain conditional U-Net vector field. Keep this baseline deliberately small.
    hidden_dims: Tuple[int, int, int] = (64, 128, 256)
    time_dim: int = 128
    num_res_blocks: int = 2
    modern: bool = True
    action_dim: int = 0
    use_lam: bool = False
    lam_ckpt: str = ""
    lam_discovery: str = ""
    warmup_steps: int = 500
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    sobolev_weight: float = 0.01
    divergence_weight: float = 0.01
    physics_dx: float = 0.0


@dataclass
class ContinuousResidualConfig:
    """Direct deterministic state-residual dynamics, separate from FM0."""

    data: str
    out: str
    resume: bool = False
    steps: int = 3000
    batch: int = 32
    # Optional memory-only gradient accumulation. The dataloader batch remains
    # the effective batch; a positive value splits it into equal microbatches.
    microbatch: int = 0
    context: int = 2
    lr: float = 3e-4
    seed: int = 0
    stats: str = ""
    checkpoint_every: int = 1000
    log_every: int = 50
    tb: int = 1
    grain_workers: int = 8
    hidden_dims: Tuple[int, int, int] = (64, 128, 256)
    num_res_blocks: int = 2
    unroll_steps: int = 1
    # Empty preserves CR0/CR1 exactly.  Non-empty enables the isolated CR1-v2
    # bounded residual output parameterization in normalized state space.
    residual_scales: Tuple[float, ...] = ()
    # CR1-v3-noise only: fixed normalized-space context-input noise.
    context_noise_sigmas: Tuple[float, ...] = ()
    # P1--P3 use physical-space endpoint losses. Zero values preserve CR0,
    # CR1, and CR1-v2 exactly.
    energy_weight: float = 0.0
    sobolev_weight: float = 0.0
    divergence_weight: float = 0.0
    physics_dx: float = 0.0
    energy_floor: float = 0.0
    gradient_ref: float = 1.0
    stable: bool = True
    warmup_steps: int = 200
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    periodic_derivatives: bool = True
