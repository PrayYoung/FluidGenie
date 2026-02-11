from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    data: str
    out: str
    batch: int = 8
    steps: int = 5000
    lr: float = 3e-4
    codebook: int = 1024
    embed: int = 64
    hidden: int = 128
    seed: int = 0
    log_every: int = 50
    tb: int = 1
    stats: str = ""
    loss_alpha: float = 1.0
    loss_beta: float = 0.25
    loss_gamma: float = 0.5


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

    codebook: int = 512
    embed: int = 64
    hidden: int = 128

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
