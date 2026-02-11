from __future__ import annotations

from dataclasses import dataclass

from configs.model_configs import DynamicsConfig, TokenizerConfig

_TOKENIZER_CODEBOOK_DEFAULT = TokenizerConfig.__dataclass_fields__["codebook"].default
_TOKENIZER_EMBED_DEFAULT = TokenizerConfig.__dataclass_fields__["embed"].default
_TOKENIZER_HIDDEN_DEFAULT = TokenizerConfig.__dataclass_fields__["hidden"].default
_TOKENIZER_STATS_DEFAULT = TokenizerConfig.__dataclass_fields__["stats"].default

_DYN_MODEL_DEFAULT = DynamicsConfig.__dataclass_fields__["model"].default
_DYN_D_MODEL_DEFAULT = DynamicsConfig.__dataclass_fields__["d_model"].default
_DYN_HEADS_DEFAULT = DynamicsConfig.__dataclass_fields__["n_heads"].default
_DYN_LAYERS_DEFAULT = DynamicsConfig.__dataclass_fields__["n_layers"].default
_DYN_DROPOUT_DEFAULT = DynamicsConfig.__dataclass_fields__["dropout"].default
_DYN_MASK_STEPS_DEFAULT = DynamicsConfig.__dataclass_fields__["mask_steps"].default


@dataclass
class DemoArgs:
    mode: str  # "tokenizer" | "rollout"
    npz: str
    out: str
    stats: str = _TOKENIZER_STATS_DEFAULT

    vq_ckpt: str = ""
    codebook: int = _TOKENIZER_CODEBOOK_DEFAULT
    embed: int = _TOKENIZER_EMBED_DEFAULT
    hidden: int = _TOKENIZER_HIDDEN_DEFAULT

    frame: int = 0
    save_gif: bool = False
    view: str = "density"  # density | vorticity | speed | channel0

    dyn_ckpt: str = ""
    start: int = 0
    horizon: int = 60
    context: int = 2

    model: str = _DYN_MODEL_DEFAULT  # transformer | maskgit
    d_model: int = _DYN_D_MODEL_DEFAULT
    n_heads: int = _DYN_HEADS_DEFAULT
    n_layers: int = _DYN_LAYERS_DEFAULT
    dropout: float = _DYN_DROPOUT_DEFAULT
    mask_steps: int = _DYN_MASK_STEPS_DEFAULT
    kv_cache: bool = True
    rollout_view: str = "density"


@dataclass
class EvalCodebookArgs:
    data: str
    vq_ckpt: str
    codebook: int = _TOKENIZER_CODEBOOK_DEFAULT
    embed: int = _TOKENIZER_EMBED_DEFAULT
    hidden: int = _TOKENIZER_HIDDEN_DEFAULT
    frames: int = 8
    episodes: int = 20
    stats: str = _TOKENIZER_STATS_DEFAULT
