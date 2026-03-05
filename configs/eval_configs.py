from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace

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
_TOKENIZER_BG_THRESH_DEFAULT = TokenizerConfig.__dataclass_fields__["bg_thresh"].default

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
_DYN_BOS_DEFAULT = DynamicsConfig.__dataclass_fields__["bos_token_id"].default
_DYN_RNG_DEFAULT = DynamicsConfig.__dataclass_fields__["seed"].default


@dataclass
class TokenizerArgs:
    stats: str = _TOKENIZER_STATS_DEFAULT
    arch: str = _TOKENIZER_ARCH_DEFAULT
    patch_size: int = _TOKENIZER_PATCH_DEFAULT
    model_dim: int = _TOKENIZER_MODEL_DIM_DEFAULT
    num_blocks: int = _TOKENIZER_BLOCKS_DEFAULT
    num_heads: int = _TOKENIZER_HEADS_DEFAULT
    dropout: float = _TOKENIZER_DROPOUT_DEFAULT
    codebook_dropout: float = _TOKENIZER_CODEBOOK_DROPOUT_DEFAULT
    bg_thresh: float = _TOKENIZER_BG_THRESH_DEFAULT
    vq_ckpt: str = ""
    codebook: int = _TOKENIZER_CODEBOOK_DEFAULT
    embed: int = _TOKENIZER_EMBED_DEFAULT
    hidden: int = _TOKENIZER_HIDDEN_DEFAULT


@dataclass
class LAMArgs:
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
class RolloutArgs:
    dyn_ckpt: str = ""
    start: int = 0
    horizon: int = 60
    context: int = 2
    model: str = _DYN_MODEL_DEFAULT  # transformer | maskgit | st_maskgit
    d_model: int = _DYN_D_MODEL_DEFAULT
    n_heads: int = _DYN_HEADS_DEFAULT
    n_layers: int = _DYN_LAYERS_DEFAULT
    dropout: float = _DYN_DROPOUT_DEFAULT
    mask_steps: int = _DYN_MASK_STEPS_DEFAULT
    kv_cache: bool = True
    view: str = "density"
    bos_token_id: int = _DYN_BOS_DEFAULT
    seed: int = _DYN_RNG_DEFAULT
    lam: LAMArgs = field(default_factory=LAMArgs)


@dataclass
class DemoArgs:
    mode: str  # "tokenizer" | "rollout"
    npz: str
    out: str
    frame: int = 0
    save_gif: bool = False
    view: str = "density"  # density | vorticity | speed | channel0
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    rollout: RolloutArgs = field(default_factory=RolloutArgs)


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


@dataclass
class RolloutConfig:
    npz_path: str
    vq_ckpt: str
    dyn_ckpt: str
    out_dir: str
    start: int = 0
    horizon: int = 60
    context: int = 2
    codebook_size: int = _TOKENIZER_CODEBOOK_DEFAULT
    embed_dim: int = _TOKENIZER_EMBED_DEFAULT
    hidden: int = _TOKENIZER_HIDDEN_DEFAULT
    d_model: int = _DYN_D_MODEL_DEFAULT
    n_heads: int = _DYN_HEADS_DEFAULT
    n_layers: int = _DYN_LAYERS_DEFAULT
    dropout: float = _DYN_DROPOUT_DEFAULT
    model_type: str = _DYN_MODEL_DEFAULT
    use_kv_cache: bool = True
    mask_steps: int = _DYN_MASK_STEPS_DEFAULT
    view: str = "density"
    stats_path: str = _TOKENIZER_STATS_DEFAULT
    tokenizer_arch: str = _TOKENIZER_ARCH_DEFAULT
    patch_size: int = _TOKENIZER_PATCH_DEFAULT
    model_dim: int = _TOKENIZER_MODEL_DIM_DEFAULT
    num_blocks: int = _TOKENIZER_BLOCKS_DEFAULT
    num_heads_tok: int = _TOKENIZER_HEADS_DEFAULT
    tokenizer_dropout: float = _TOKENIZER_DROPOUT_DEFAULT
    codebook_dropout: float = _TOKENIZER_CODEBOOK_DROPOUT_DEFAULT
    bg_thresh: float = _TOKENIZER_BG_THRESH_DEFAULT
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
    bos_token_id: int = _DYN_BOS_DEFAULT
    seed: int = _DYN_RNG_DEFAULT


def rollout_config_from_demo(args: DemoArgs) -> RolloutConfig:
    return RolloutConfig(
        npz_path=args.npz,
        vq_ckpt=args.tokenizer.vq_ckpt,
        dyn_ckpt=args.rollout.dyn_ckpt,
        out_dir=args.out,
        start=args.rollout.start,
        horizon=args.rollout.horizon,
        context=args.rollout.context,
        codebook_size=args.tokenizer.codebook,
        embed_dim=args.tokenizer.embed,
        hidden=args.tokenizer.hidden,
        d_model=args.rollout.d_model,
        n_heads=args.rollout.n_heads,
        n_layers=args.rollout.n_layers,
        dropout=args.rollout.dropout,
        model_type=args.rollout.model,
        use_kv_cache=args.rollout.kv_cache,
        mask_steps=args.rollout.mask_steps,
        view=args.rollout.view,
        stats_path=args.tokenizer.stats,
        tokenizer_arch=args.tokenizer.arch,
        patch_size=args.tokenizer.patch_size,
        model_dim=args.tokenizer.model_dim,
        num_blocks=args.tokenizer.num_blocks,
        num_heads_tok=args.tokenizer.num_heads,
        tokenizer_dropout=args.tokenizer.dropout,
        codebook_dropout=args.tokenizer.codebook_dropout,
        bg_thresh=args.tokenizer.bg_thresh,
        bos_token_id=args.rollout.bos_token_id,
        seed=args.rollout.seed,
        use_lam=args.rollout.lam.use_lam,
        lam_ckpt=args.rollout.lam.lam_ckpt,
        lam_model_dim=args.rollout.lam.lam_model_dim,
        lam_latent_dim=args.rollout.lam.lam_latent_dim,
        lam_num_latents=args.rollout.lam.lam_num_latents,
        lam_patch_size=args.rollout.lam.lam_patch_size,
        lam_num_blocks=args.rollout.lam.lam_num_blocks,
        lam_num_heads=args.rollout.lam.lam_num_heads,
        lam_dropout=args.rollout.lam.lam_dropout,
        lam_codebook_dropout=args.rollout.lam.lam_codebook_dropout,
    )


def tokenizer_recon_kwargs_from_demo(args: DemoArgs) -> dict:
    return {
        "npz_path": args.npz,
        "vq_ckpt": args.tokenizer.vq_ckpt,
        "out_dir": args.out,
        "frame": args.frame,
        "codebook_size": args.tokenizer.codebook,
        "embed_dim": args.tokenizer.embed,
        "hidden": args.tokenizer.hidden,
        "stats_path": args.tokenizer.stats if args.tokenizer.stats else None,
        "save_gif": args.save_gif,
        "view": args.view,
        "tokenizer_arch": args.tokenizer.arch,
        "patch_size": args.tokenizer.patch_size,
        "model_dim": args.tokenizer.model_dim,
        "num_blocks": args.tokenizer.num_blocks,
        "num_heads": args.tokenizer.num_heads,
        "dropout": args.tokenizer.dropout,
        "codebook_dropout": args.tokenizer.codebook_dropout,
        "bg_thresh": args.tokenizer.bg_thresh,
    }


def apply_ckpt_config_to_demo(args: DemoArgs) -> DemoArgs:
    """
    Fill DemoArgs from config.json saved next to ckpts.
    Priority: CLI values > ckpt config > defaults.
    """
    defaults = DemoArgs(mode=args.mode, npz=args.npz, out=args.out)

    # tokenizer config
    tok_cfg = load_config_json(args.tokenizer.vq_ckpt)
    if tok_cfg and isinstance(tok_cfg.get("config"), dict):
        tok_defaults = defaults.tokenizer
        tok = merge_dataclass_from_config(args.tokenizer, tok_cfg["config"], tok_defaults)
    else:
        tok = args.tokenizer

    # dynamics config
    dyn_cfg = load_config_json(args.rollout.dyn_ckpt)
    if dyn_cfg and isinstance(dyn_cfg.get("config"), dict):
        roll_defaults = defaults.rollout
        roll = merge_dataclass_from_config(args.rollout, dyn_cfg["config"], roll_defaults)
    else:
        roll = args.rollout

    # LAM config
    if isinstance(roll.lam, dict):
        roll = replace(roll, lam=LAMArgs(**roll.lam))
    lam_cfg = load_config_json(roll.lam.lam_ckpt)
    if lam_cfg and isinstance(lam_cfg.get("config"), dict):
        lam_defaults = defaults.rollout.lam
        lam = merge_dataclass_from_config(roll.lam, lam_cfg["config"], lam_defaults)
    else:
        lam = roll.lam

    roll = replace(roll, lam=lam)
    return replace(args, tokenizer=tok, rollout=roll)


def apply_ckpt_config_to_rollout(cfg: RolloutConfig) -> RolloutConfig:
    """
    Fill RolloutConfig from config.json saved next to ckpts.
    Priority: CLI values > ckpt config > defaults.
    """
    defaults = RolloutConfig(
        npz_path=cfg.npz_path,
        vq_ckpt=cfg.vq_ckpt,
        dyn_ckpt=cfg.dyn_ckpt,
        out_dir=cfg.out_dir,
        start=cfg.start,
        horizon=cfg.horizon,
        context=cfg.context,
    )

    def _maybe_set(obj: RolloutConfig, field: str, value):
        if value is None:
            return obj
        if getattr(obj, field) == getattr(defaults, field):
            return replace(obj, **{field: value})
        return obj

    tok_cfg = load_config_json(cfg.vq_ckpt)
    if tok_cfg and isinstance(tok_cfg.get("config"), dict):
        t = tok_cfg["config"]
        cfg = _maybe_set(cfg, "codebook_size", t.get("codebook"))
        cfg = _maybe_set(cfg, "embed_dim", t.get("embed"))
        cfg = _maybe_set(cfg, "hidden", t.get("hidden"))
        cfg = _maybe_set(cfg, "tokenizer_arch", t.get("arch"))
        cfg = _maybe_set(cfg, "patch_size", t.get("patch_size"))
        cfg = _maybe_set(cfg, "model_dim", t.get("model_dim"))
        cfg = _maybe_set(cfg, "num_blocks", t.get("num_blocks"))
        cfg = _maybe_set(cfg, "num_heads_tok", t.get("num_heads"))
        cfg = _maybe_set(cfg, "tokenizer_dropout", t.get("dropout"))
        cfg = _maybe_set(cfg, "codebook_dropout", t.get("codebook_dropout"))
        cfg = _maybe_set(cfg, "bg_thresh", t.get("bg_thresh"))
        cfg = _maybe_set(cfg, "stats_path", t.get("stats"))

    dyn_cfg = load_config_json(cfg.dyn_ckpt)
    if dyn_cfg and isinstance(dyn_cfg.get("config"), dict):
        d = dyn_cfg["config"]
        cfg = _maybe_set(cfg, "d_model", d.get("d_model"))
        cfg = _maybe_set(cfg, "n_heads", d.get("n_heads"))
        cfg = _maybe_set(cfg, "n_layers", d.get("n_layers"))
        cfg = _maybe_set(cfg, "dropout", d.get("dropout"))
        cfg = _maybe_set(cfg, "model_type", d.get("model"))
        cfg = _maybe_set(cfg, "mask_steps", d.get("mask_steps"))
        cfg = _maybe_set(cfg, "bos_token_id", d.get("bos_token_id"))
        cfg = _maybe_set(cfg, "seed", d.get("seed"))
        cfg = _maybe_set(cfg, "use_lam", d.get("use_lam"))
        cfg = _maybe_set(cfg, "lam_ckpt", d.get("lam_ckpt"))
        cfg = _maybe_set(cfg, "lam_model_dim", d.get("lam_model_dim"))
        cfg = _maybe_set(cfg, "lam_latent_dim", d.get("lam_latent_dim"))
        cfg = _maybe_set(cfg, "lam_num_latents", d.get("lam_num_latents"))
        cfg = _maybe_set(cfg, "lam_patch_size", d.get("lam_patch_size"))
        cfg = _maybe_set(cfg, "lam_num_blocks", d.get("lam_num_blocks"))
        cfg = _maybe_set(cfg, "lam_num_heads", d.get("lam_num_heads"))
        cfg = _maybe_set(cfg, "lam_dropout", d.get("lam_dropout"))
        cfg = _maybe_set(cfg, "lam_codebook_dropout", d.get("lam_codebook_dropout"))
        cfg = _maybe_set(cfg, "stats_path", d.get("stats"))
        cfg = _maybe_set(cfg, "patch_size", d.get("tok_patch_size"))
        cfg = _maybe_set(cfg, "model_dim", d.get("tok_model_dim"))
        cfg = _maybe_set(cfg, "num_blocks", d.get("tok_num_blocks"))
        cfg = _maybe_set(cfg, "num_heads_tok", d.get("tok_num_heads"))
        cfg = _maybe_set(cfg, "tokenizer_dropout", d.get("tok_dropout"))
        cfg = _maybe_set(cfg, "codebook_dropout", d.get("tok_codebook_dropout"))
        cfg = _maybe_set(cfg, "bg_thresh", d.get("bg_thresh"))

    lam_cfg = load_config_json(cfg.lam_ckpt)
    if lam_cfg and isinstance(lam_cfg.get("config"), dict):
        l = lam_cfg["config"]
        cfg = _maybe_set(cfg, "lam_model_dim", l.get("model_dim"))
        cfg = _maybe_set(cfg, "lam_latent_dim", l.get("latent_dim"))
        cfg = _maybe_set(cfg, "lam_num_latents", l.get("num_latents"))
        cfg = _maybe_set(cfg, "lam_patch_size", l.get("patch_size"))
        cfg = _maybe_set(cfg, "lam_num_blocks", l.get("num_blocks"))
        cfg = _maybe_set(cfg, "lam_num_heads", l.get("num_heads"))
        cfg = _maybe_set(cfg, "lam_dropout", l.get("dropout"))
        cfg = _maybe_set(cfg, "lam_codebook_dropout", l.get("codebook_dropout"))
        cfg = _maybe_set(cfg, "stats_path", l.get("stats"))

    return cfg
from configs.config_io import load_config_json, merge_dataclass_from_config
