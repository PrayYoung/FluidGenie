"""Train the isolated FM0 direct state-space conditional flow-matching baseline."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np

import jax
import jax.numpy as jnp
import optax
import tyro
from flax.training import train_state
from tqdm import trange

from fluidgenie.training.optimizers import make_optimizer
from fluidgenie.training.continuous_residual_physics_losses import periodic_derivative
from fluidgenie.training.run_metadata import prepare_run
from configs.model_configs import FlowMatchingConfig
from fluidgenie.data.dataset_npz import create_grain_dataloader
from fluidgenie.models.lam_conditioning import load_frozen_lam
from fluidgenie.models.flow_matching import ConditionalFlowUNet
from fluidgenie.training.checkpoint_utils import (
    save_params,
    save_training_state,
    restore_training_state,
)
from fluidgenie.training.logging_utils import TrainingLogger


class TrainState(train_state.TrainState):
    pass


def flow_matching_loss(
    apply_fn,
    params,
    context,
    target,
    rng,
    action_cond=None,
    physics_weights=(0.0, 0.0),
    physical_scale=None,
    dx=1.0 / 64,
):
    """Conditional flow matching from N(0,I) to the next normalized field."""
    noise_key, time_key = jax.random.split(rng)
    noise = jax.random.normal(noise_key, target.shape, dtype=target.dtype)
    tau = jax.random.uniform(
        time_key, (target.shape[0],), minval=1e-5, maxval=1.0 - 1e-5, dtype=target.dtype
    )
    tau_field = tau[:, None, None, None]
    x_tau = (1.0 - tau_field) * noise + tau_field * target
    target_velocity = target - noise
    predicted_velocity = apply_fn(
        {"params": params},
        x_tau,
        tau,
        context,
        training=True,
        **({"action_cond": action_cond} if action_cond is not None else {}),
    )
    loss = jnp.mean((predicted_velocity - target_velocity) ** 2)
    data_loss = loss
    endpoint_estimate = x_tau + (1 - tau_field) * predicted_velocity
    endpoint_error = endpoint_estimate - target
    if physical_scale is not None:
        endpoint_error = endpoint_error * physical_scale
    du_dx = periodic_derivative(endpoint_error[..., 0], 2, dx)
    dv_dy = periodic_derivative(endpoint_error[..., 1], 1, dx)
    sobolev = jnp.mean(
        du_dx**2
        + dv_dy**2
        + periodic_derivative(endpoint_error[..., 0], 1, dx) ** 2
        + periodic_derivative(endpoint_error[..., 1], 2, dx) ** 2
    )
    physical_endpoint = (
        endpoint_estimate
        if physical_scale is None
        else endpoint_estimate * physical_scale
    )
    divergence = jnp.mean(
        (
            periodic_derivative(physical_endpoint[..., 0], 2, dx)
            + periodic_derivative(physical_endpoint[..., 1], 1, dx)
        )
        ** 2
    )
    loss = loss + physics_weights[0] * sobolev + physics_weights[1] * divergence
    return loss, {
        "loss": loss,
        "data_loss": data_loss,
        "sobolev_loss": sobolev,
        "divergence_loss": divergence,
        "tau_mean": jnp.mean(tau),
    }


def make_train_step(physics_weights=(0.0, 0.0), physical_scale=None, dx=1.0 / 64):
    @jax.jit
    def step(state, context, target, rng, action_cond=None):
        def loss_fn(params):
            return flow_matching_loss(
                state.apply_fn,
                params,
                context,
                target,
                rng,
                action_cond,
                physics_weights,
                physical_scale,
                dx,
            )

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), {
            **metrics,
            "grad_norm": optax.global_norm(grads),
        }

    return step


train_step = make_train_step()


def main() -> None:
    args = tyro.cli(FlowMatchingConfig)
    if args.context != 2:
        raise ValueError("FM0 intentionally fixes context=2; use --context 2")

    loader = create_grain_dataloader(
        args.data,
        batch_size=args.batch,
        context=args.context,
        pred=1,
        seed=args.seed,
        num_workers=args.grain_workers,
        stats_path=args.stats or None,
        action_dim=0 if args.use_lam else args.action_dim,
    )
    data_iter = iter(loader)
    initial = next(data_iter)
    context0, target0 = initial[:2]
    _, _, h, w, channels = context0.shape
    if h % 4 or w % 4:
        raise ValueError(f"FM0 requires H,W divisible by 4, got {h}x{w}")

    lam_encode = None
    initial_action = (
        jnp.asarray(initial[2]) if args.action_dim and not args.use_lam else None
    )
    if args.use_lam:
        if not args.lam_discovery:
            raise ValueError(
                "--lam-discovery is required to certify semantic LAM conditioning"
            )
        discovery = json.loads(Path(args.lam_discovery).read_text())
        if not discovery.get("discovery_gate_passed") or not discovery.get(
            "five_action_mapping_gate"
        ):
            raise ValueError(
                "LAM discovery and five-action mapping gates must pass before downstream training"
            )
        if not args.lam_ckpt:
            raise ValueError("--lam-ckpt is required with --use-lam")
        from fluidgenie.eval.action_discovery import require_discovery_certificate

        require_discovery_certificate(discovery)
        protocol = discovery.get("protocol", {})
        if (
            Path(protocol.get("checkpoint", "")).resolve()
            != Path(args.lam_ckpt).resolve()
            or Path(protocol.get("calibration_dir", "")).resolve()
            != Path(args.data).resolve()
        ):
            raise ValueError(
                "Discovery certification does not match this frozen LAM and calibration split"
            )
        _, _, lam_encode, lam_cfg = load_frozen_lam(
            args.lam_ckpt,
            jnp.concatenate([jnp.asarray(context0), jnp.asarray(target0)], axis=1),
        )

        from fluidgenie.eval.utils import resolve_stats_path

        lam_stats = resolve_stats_path("", args.lam_ckpt)
        if not args.stats or not lam_stats:
            raise ValueError("Flow and LAM require training normalization statistics")
        left, right = np.load(args.stats), np.load(lam_stats)
        if set(left.files) != set(right.files) or any(
            not np.array_equal(left[k], right[k]) for k in left.files
        ):
            raise ValueError(
                "Flow and frozen LAM must use identical training normalization"
            )
        args.action_dim = int(lam_cfg["latent_dim"])
        initial_action = lam_encode(
            jnp.concatenate([jnp.asarray(context0), jnp.asarray(target0)], axis=1)
        )
    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    model = ConditionalFlowUNet(
        channels=channels,
        context_frames=args.context,
        hidden_dims=args.hidden_dims,
        time_dim=args.time_dim,
        num_res_blocks=args.num_res_blocks,
        modern=args.modern,
        action_dim=args.action_dim,
    )
    params = model.init(
        init_key,
        jnp.asarray(target0[:, 0]),
        jnp.zeros((target0.shape[0],), dtype=jnp.float32),
        jnp.asarray(context0),
        training=True,
        **({"action_cond": initial_action} if args.action_dim else {}),
    )["params"]
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=make_optimizer(
            args.lr,
            args.steps,
            args.warmup_steps,
            args.weight_decay,
            args.max_grad_norm,
        )
        if args.modern
        else optax.adam(args.lr),
    )

    if args.physics_dx == 0:
        args.physics_dx = 1.0 / context0.shape[3]
    if not args.modern and (args.sobolev_weight or args.divergence_weight):
        raise ValueError(
            "Legacy FM physics penalties require explicit physical scaling; use --modern or disable auxiliary weights"
        )
    scale = None
    if args.modern:
        if not args.stats:
            raise ValueError(
                "Corrected FM training requires training-only z-score statistics"
            )
        stats = np.load(args.stats)
        if "min" in stats or "max" in stats:
            raise ValueError("Use compute_stats --zscore-only for corrected FM")
        scale = jnp.asarray(stats["std"] + 1e-6)
    step_fn = make_train_step(
        (args.sobolev_weight, args.divergence_weight), scale, args.physics_dx
    )
    out = prepare_run(args.out, "flow_matching_fm0", args, resume=args.resume)
    if args.use_lam:
        import shutil

        destination = out / "action_calibration.json"
        if Path(args.lam_discovery).resolve() != destination.resolve():
            shutil.copy2(args.lam_discovery, destination)
    if args.modern:
        np.savez(out / "stats.npz", mean=stats["mean"], std=stats["std"])
    logger = TrainingLogger(out, "flow_matching", args.log_every, bool(args.tb))
    if args.resume:
        state, rng = restore_training_state(out, state, rng, data_iter)
    try:
        for step in trange(int(state.step), args.steps):
            batch = next(data_iter)
            context, target = batch[:2]
            rng, step_key = jax.random.split(rng)
            state, metrics = step_fn(
                state,
                jnp.asarray(context),
                jnp.asarray(target[:, 0]),
                step_key,
                lam_encode(
                    jnp.concatenate([jnp.asarray(context), jnp.asarray(target)], axis=1)
                )
                if args.use_lam
                else (jnp.asarray(batch[2]) if args.action_dim else None),
            )
            if logger.should_log(step):
                logger.log(step, metrics)
            if (
                args.checkpoint_every > 0
                and step > 0
                and step % args.checkpoint_every == 0
            ):
                save_params(out, f"step_{step:06d}", state.params)
                save_params(out, "latest", state.params)
                save_training_state(out, state, rng)
    finally:
        logger.close()

    save_training_state(out, state, rng)
    save_params(out, "final", state.params)
    save_params(out, "latest", state.params)
    print(f"Saved FM0 flow-matching checkpoint to {out}")


if __name__ == "__main__":
    main()
