from __future__ import annotations

from dataclasses import dataclass

from configs.model_configs import DynamicsConfig, TokenizerConfig

_TOKENIZER_CODEBOOK_DEFAULT = TokenizerConfig.__dataclass_fields__["codebook"].default
_TOKENIZER_EMBED_DEFAULT = TokenizerConfig.__dataclass_fields__["embed"].default
_TOKENIZER_HIDDEN_DEFAULT = TokenizerConfig.__dataclass_fields__["hidden"].default
_TOKENIZER_STATS_DEFAULT = TokenizerConfig.__dataclass_fields__["stats"].default
_TOKENIZER_ARCH_DEFAULT = TokenizerConfig.__dataclass_fields__["arch"].default
_TOKENIZER_PATCH_DEFAULT = TokenizerConfig.__dataclass_fields__["patch_size"].default
_TOKENIZER_MODEL_DIM_DEFAULT = TokenizerConfig.__dataclass_fields__["model_dim"].default
_TOKENIZER_BLOCKS_DEFAULT = TokenizerConfig.__dataclass_fields__["num_blocks"].default
_TOKENIZER_HEADS_DEFAULT = TokenizerConfig.__dataclass_fields__["num_heads"].default
_TOKENIZER_DROPOUT_DEFAULT = TokenizerConfig.__dataclass_fields__["dropout"].default
_TOKENIZER_CODEBOOK_DROPOUT_DEFAULT = TokenizerConfig.__dataclass_fields__["codebook_dropout"].default

_DYN_MODEL_DEFAULT = DynamicsConfig.__dataclass_fields__["model"].default
_DYN_D_MODEL_DEFAULT = DynamicsConfig.__dataclass_fields__["d_model"].default
_DYN_HEADS_DEFAULT = DynamicsConfig.__dataclass_fields__["n_heads"].default
_DYN_LAYERS_DEFAULT = DynamicsConfig.__dataclass_fields__["n_layers"].default
_DYN_DROPOUT_DEFAULT = DynamicsConfig.__dataclass_fields__["dropout"].default
_DYN_MASK_STEPS_DEFAULT = DynamicsConfig.__dataclass_fields__["mask_steps"].default
_DYN_USE_LAM_DEFAULT = DynamicsConfig.__dataclass_fields__["use_lam"].default
_DYN_LAM_CKPT_DEFAULT = DynamicsConfig.__dataclass_fields__["lam_ckpt"].default
_DYN_LAM_MODEL_DIM_DEFAULT = DynamicsConfig.__dataclass_fields__["lam_model_dim"].default
_DYN_LAM_LATENT_DIM_DEFAULT = DynamicsConfig.__dataclass_fields__["lam_latent_dim"].default
_DYN_LAM_NUM_LATENTS_DEFAULT = DynamicsConfig.__dataclass_fields__["lam_num_latents"].default
_DYN_LAM_PATCH_SIZE_DEFAULT = DynamicsConfig.__dataclass_fields__["lam_patch_size"].default
_DYN_LAM_NUM_BLOCKS_DEFAULT = DynamicsConfig.__dataclass_fields__["lam_num_blocks"].default
_DYN_LAM_NUM_HEADS_DEFAULT = DynamicsConfig.__dataclass_fields__["lam_num_heads"].default
_DYN_LAM_DROPOUT_DEFAULT = DynamicsConfig.__dataclass_fields__["lam_dropout"].default
_DYN_LAM_CODEBOOK_DROPOUT_DEFAULT = DynamicsConfig.__dataclass_fields__["lam_codebook_dropout"].default


@dataclass
class DemoArgs:
    mode: str  # "tokenizer" | "rollout"
    npz: str
    out: str
    stats: str = _TOKENIZER_STATS_DEFAULT
    tokenizer_arch: str = _TOKENIZER_ARCH_DEFAULT
    patch_size: int = _TOKENIZER_PATCH_DEFAULT
    model_dim: int = _TOKENIZER_MODEL_DIM_DEFAULT
    num_blocks: int = _TOKENIZER_BLOCKS_DEFAULT
    num_heads: int = _TOKENIZER_HEADS_DEFAULT
    dropout: float = _TOKENIZER_DROPOUT_DEFAULT
    codebook_dropout: float = _TOKENIZER_CODEBOOK_DROPOUT_DEFAULT

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
    bos_token_id: int = 0
    rng_seed: int = 0

    use_lam: bool = _DYN_USE_LAM_DEFAULT
    lam_ckpt: str = _DYN_LAM_CKPT_DEFAULT
    lam_model_dim: int = _DYN_LAM_MODEL_DIM_DEFAULT
    lam_latent_dim: int = _DYN_LAM_LATENT_DIM_DEFAULT
    lam_num_latents: int = _DYN_LAM_NUM_LATENTS_DEFAULT
    lam_patch_size: int = _DYN_LAM_PATCH_SIZE_DEFAULT
    lam_num_blocks: int = _DYN_LAM_NUM_BLOCKS_DEFAULT
    lam_num_heads: int = _DYN_LAM_NUM_HEADS_DEFAULT
    lam_dropout: float = _DYN_LAM_DROPOUT_DEFAULT
    lam_codebook_dropout: float = _DYN_LAM_CODEBOOK_DROPOUT_DEFAULT


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
    seed: int = 0
    tokenizer_arch: str = _TOKENIZER_ARCH_DEFAULT
    patch_size: int = _TOKENIZER_PATCH_DEFAULT
    model_dim: int = _TOKENIZER_MODEL_DIM_DEFAULT
    num_blocks: int = _TOKENIZER_BLOCKS_DEFAULT
    num_heads: int = _TOKENIZER_HEADS_DEFAULT
    dropout: float = _TOKENIZER_DROPOUT_DEFAULT
    codebook_dropout: float = _TOKENIZER_CODEBOOK_DROPOUT_DEFAULT
