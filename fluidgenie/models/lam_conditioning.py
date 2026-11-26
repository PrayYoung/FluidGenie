"""Load a frozen, video-only LAM for continuous dynamics conditioning."""

import jax
import jax.numpy as jnp
from configs.config_io import load_config_json
from fluidgenie.models.lam import LatentActionModel
from fluidgenie.training.checkpoint_utils import load_params


def load_frozen_lam(checkpoint, example_video):
    saved = load_config_json(checkpoint)
    if not saved or saved.get("kind") != "lam":
        raise ValueError("A LAM checkpoint with configuration is required")
    cfg = saved["config"]
    if (cfg.get("architecture_version") or 0) < 2:
        raise ValueError("A causal architecture_version >= 2 LAM is required")

    def validate_video(video):
        if not 2 <= video.shape[1] <= int(cfg["seq_len"]):
            raise ValueError(
                "LAM input length must be between 2 and the trained seq_len"
            )

    validate_video(example_video)
    model = LatentActionModel(
        in_dim=example_video.shape[-1],
        architecture_version=cfg.get("architecture_version", 2),
        **{
            k: cfg[k]
            for k in (
                "model_dim",
                "latent_dim",
                "num_latents",
                "patch_size",
                "num_blocks",
                "num_heads",
                "dropout",
                "codebook_dropout",
            )
        },
    )
    initial = model.init(
        jax.random.PRNGKey(0), {"videos": example_video}, training=False
    )["params"]
    params = load_params(checkpoint, initial)

    @jax.jit
    def encoded(video):
        return jax.lax.stop_gradient(
            model.apply(
                {"params": params}, video, training=False, method=model.vq_encode
            )["z_q"][:, -1, 0].astype(example_video.dtype)
        )

    def encode(video):
        validate_video(video)
        return encoded(video)

    return model, params, encode, cfg
