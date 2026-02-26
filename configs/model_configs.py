from __future__ import annotations

from dataclasses import dataclass

# Shared defaults to keep tokenizer/dynamics consistent
DEFAULT_CODEBOOK = 512
DEFAULT_EMBED = 64
DEFAULT_HIDDEN = 128
DEFAULT_PATCH_SIZE = 4


@dataclass
class TokenizerConfig:
    data: str
    out: str
    batch: int = 128
    steps: int = 10000
    lr: float = 3e-4
    codebook: int = DEFAULT_CODEBOOK
    embed: int = DEFAULT_EMBED
    hidden: int = DEFAULT_HIDDEN
    seed: int = 0
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
    seq_len: int = 2
    grain_workers: int = 8


@dataclass
class DynamicsConfig:
    data: str
    vq_ckpt: str
    out: str
    steps: int = 20000
    batch: int = 4
    context: int = 2
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
    log_every: int = 50
    tb: int = 1
    stats: str = ""

    # MaskGIT options (only used when model = "maskgit")
    mask_ratio_min: float = 0.1
    mask_ratio_max: float = 0.9
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


@dataclass
class LAMConfig:
    data: str
    out: str
    steps: int = 20000
    batch: int = 4
    seq_len: int = 8
    lr: float = 3e-4
    seed: int = 0
    log_every: int = 50
    tb: int = 1
    stats: str = ""

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
