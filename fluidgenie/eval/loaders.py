from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from fluidgenie.models.base_tokenizer import VQConfig
from fluidgenie.models.base_dynamics import DynConfig
from fluidgenie.models.dynamics_st import DynamicsSTMaskGIT
from fluidgenie.models.lam import LatentActionModel
from fluidgenie.eval.utils import (
    load_tokenizer_params,
    load_dyn_params,
    get_codebook_and_decoder_params,
    make_vq_encode_tokens,
    make_st_encode_tokens,
)
from fluidgenie.training.checkpoint_utils import load_params
from configs.eval_configs import RolloutConfig


def load_tokenizer(cfg: RolloutConfig, H: int, W: int, C: int):
    vq_cfg = VQConfig(codebook_size=cfg.codebook_size, embed_dim=cfg.embed_dim, hidden=cfg.hidden)
    model, params = load_tokenizer_params(
        cfg.tokenizer_arch,
        vq_cfg,
        in_channels=C,
        H=H,
        W=W,
        ckpt_path=cfg.vq_ckpt,
        patch_size=cfg.patch_size,
        model_dim=cfg.model_dim,
        num_blocks=cfg.num_blocks,
        num_heads=cfg.num_heads_tok if cfg.num_heads_tok is not None else cfg.n_heads,
        dropout=cfg.tokenizer_dropout,
        codebook_dropout=cfg.codebook_dropout,
        bg_thresh=cfg.bg_thresh,
    )
    if cfg.tokenizer_arch == "st":
        encode_fn = make_st_encode_tokens(model)
        codebook = None
        dec_params = None
    else:
        encode_fn = make_vq_encode_tokens(model)
        codebook, dec_params = get_codebook_and_decoder_params(params)
    return model, params, encode_fn, vq_cfg, codebook, dec_params


def load_dynamics_model(cfg: RolloutConfig, max_len: int, rng: jax.Array):
    if cfg.model_type == "st_maskgit":
        rng, mask_rng = jax.random.split(rng)
        model = DynamicsSTMaskGIT(
            model_dim=cfg.d_model,
            num_latents=cfg.codebook_size,
            num_blocks=cfg.n_layers,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            mask_ratio_min=0.0,
            mask_ratio_max=1.0,
        )
        tok_seq0 = jnp.zeros((1, cfg.context + 1, 1, 1), dtype=jnp.int32)
        init_batch = {"video_tokens": tok_seq0, "mask_rng": mask_rng}
        if cfg.use_lam:
            init_batch["latent_actions"] = jnp.zeros((1, cfg.context, 1, cfg.lam_latent_dim), dtype=jnp.float32)
        params_init = model.init(rng, init_batch, training=False)["params"]
        params = load_params(cfg.dyn_ckpt, params_init)
        return model, params

    vocab_size = cfg.codebook_size + (1 if cfg.model_type == "maskgit" else 0)
    dyn_cfg = DynConfig(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        max_len=max_len,
    )
    model, params = load_dyn_params(dyn_cfg, max_len=max_len, ckpt_path=cfg.dyn_ckpt)
    return model, params


def load_lam_model(cfg: RolloutConfig, H: int, W: int, C: int, rng: jax.Array):
    if not cfg.use_lam:
        return None, None
    if not cfg.lam_ckpt:
        raise ValueError("--lam_ckpt is required when use_lam=True")
    lam_model = LatentActionModel(
        in_dim=C,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.lam_num_latents,
        patch_size=cfg.lam_patch_size,
        num_blocks=cfg.lam_num_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=cfg.lam_dropout,
        codebook_dropout=cfg.lam_codebook_dropout,
    )
    lam_init = lam_model.init(
        rng,
        {"videos": jnp.zeros((1, cfg.context, H, W, C), dtype=jnp.float32)},
        training=False,
    )["params"]
    lam_params = load_params(cfg.lam_ckpt, lam_init)
    return lam_model, lam_params
