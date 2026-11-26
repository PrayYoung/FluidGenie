"""Train CR0/CR1 deterministic direct state-space residual dynamics."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training import train_state
from tqdm import trange

from fluidgenie.training.optimizers import make_optimizer
from fluidgenie.training.run_metadata import prepare_run
from configs.model_configs import ContinuousResidualConfig
from fluidgenie.data.dataset_npz import create_grain_dataloader
from fluidgenie.models.continuous_residual import ContinuousResidualUNet
from fluidgenie.training.checkpoint_utils import (
    save_params,
    save_training_state,
    restore_training_state,
)
from fluidgenie.training.logging_utils import TrainingLogger
from fluidgenie.training.continuous_residual_physics_losses import (
    PhysicsLossConfig,
    endpoint_physics_losses,
)


class TrainState(train_state.TrainState):
    pass


def residual_unroll_loss(
    apply_fn,
    params,
    context,
    targets,
    noise_key=None,
    context_noise_sigmas=(),
    physics: PhysicsLossConfig | None = None,
    minimum=None,
    span=None,
    capture_saturation=False,
):
    """Roll out predictions in-context; CR0 is the special case K=1."""
    saturation = jnp.asarray(0.0, dtype=context.dtype)
    current_context = context
    loss = jnp.asarray(0.0, dtype=context.dtype)
    energy_loss = jnp.asarray(0.0, dtype=context.dtype)
    sobolev_loss = jnp.asarray(0.0, dtype=context.dtype)
    divergence_loss = jnp.asarray(0.0, dtype=context.dtype)
    noise_sigmas = (
        jnp.asarray(context_noise_sigmas, dtype=context.dtype)
        if context_noise_sigmas
        else None
    )
    for index in range(targets.shape[1]):
        if noise_sigmas is None:
            model_context = current_context
        else:
            noise_key, step_key = jax.random.split(noise_key)
            noise = jax.random.normal(
                step_key, current_context.shape, dtype=context.dtype
            )
            model_context = current_context + noise * noise_sigmas.reshape(
                1, 1, 1, 1, -1
            )
        if capture_saturation:
            delta, diagnostics = apply_fn(
                {"params": params},
                model_context,
                training=True,
                mutable=["diagnostics"],
            )
            saturation = (
                saturation
                + diagnostics.get("diagnostics", {}).get(
                    "tanh_sat_frac", (jnp.asarray(0.0),)
                )[-1]
            )
        else:
            delta = apply_fn({"params": params}, model_context, training=True)
        prediction = current_context[:, -1] + delta
        target = targets[:, index]
        loss = loss + jnp.mean((prediction - target) ** 2)
        if physics is not None and physics.enabled:
            physics_losses = endpoint_physics_losses(
                prediction, target, minimum, span, physics
            )
            energy_loss = energy_loss + physics_losses["energy"]
            sobolev_loss = sobolev_loss + physics_losses["sobolev"]
            divergence_loss = divergence_loss + physics_losses["divergence"]
        current_context = jnp.concatenate(
            [current_context[:, 1:], prediction[:, None]], axis=1
        )
    loss = loss / targets.shape[1]
    if physics is None or not physics.enabled:
        return loss, {
            "loss": loss,
            "unroll_steps": jnp.asarray(targets.shape[1], dtype=jnp.float32),
            **(
                {"tanh_sat_frac": saturation / targets.shape[1]}
                if capture_saturation
                else {}
            ),
        }
    energy_loss = energy_loss / targets.shape[1]
    sobolev_loss = sobolev_loss / targets.shape[1]
    divergence_loss = divergence_loss / targets.shape[1]
    total = (
        loss
        + physics.energy_weight * energy_loss
        + physics.sobolev_weight * sobolev_loss
        + physics.divergence_weight * divergence_loss
    )
    return total, {
        "loss": total,
        "data_loss": loss,
        "energy_loss": energy_loss,
        "sobolev_loss": sobolev_loss,
        "divergence_loss": divergence_loss,
        "unroll_steps": jnp.asarray(targets.shape[1], dtype=jnp.float32),
        **(
            {"tanh_sat_frac": saturation / targets.shape[1]}
            if capture_saturation
            else {}
        ),
    }


def make_train_step(context_noise_sigmas=(), capture_saturation=False):
    @jax.jit
    def train_step(state: TrainState, context, targets, noise_key):
        def loss_fn(params):
            return residual_unroll_loss(
                state.apply_fn,
                params,
                context,
                targets,
                noise_key,
                context_noise_sigmas,
                capture_saturation=capture_saturation,
            )

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        metrics = {**metrics, "grad_norm": optax.global_norm(grads)}
        return state.apply_gradients(grads=grads), metrics

    return train_step


def make_physics_train_step(
    physics: PhysicsLossConfig, minimum: jnp.ndarray, span: jnp.ndarray
):
    """P1/P2/P3 path; only one fixed auxiliary coefficient may be non-zero."""

    @jax.jit
    def train_step(state: TrainState, context, targets):
        def loss_fn(params):
            return residual_unroll_loss(
                state.apply_fn,
                params,
                context,
                targets,
                physics=physics,
                minimum=minimum,
                span=span,
                capture_saturation=True,
            )

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        metrics = {**metrics, "grad_norm": optax.global_norm(grads)}
        return state.apply_gradients(grads=grads), metrics

    return train_step


def make_microbatch_train_step(
    microbatch: int, context_noise_sigmas=(), capture_saturation=False
):
    """Accumulate mean microbatch gradients and apply one effective-batch update."""

    @jax.jit
    def train_step(state: TrainState, context, targets, noise_key):
        count = context.shape[0] // microbatch

        def micro_loss(params, micro_context, micro_targets, micro_key):
            return residual_unroll_loss(
                state.apply_fn,
                params,
                micro_context,
                micro_targets,
                micro_key,
                context_noise_sigmas,
                capture_saturation=capture_saturation,
            )

        def body(index, carry):
            grad_sum, loss_sum, saturation_sum = carry
            start = index * microbatch
            micro_context = jax.lax.dynamic_slice_in_dim(
                context, start, microbatch, axis=0
            )
            micro_targets = jax.lax.dynamic_slice_in_dim(
                targets, start, microbatch, axis=0
            )
            micro_key = jax.random.fold_in(noise_key, index)
            ((loss, aux), grads) = jax.value_and_grad(micro_loss, has_aux=True)(
                state.params, micro_context, micro_targets, micro_key
            )
            return (
                jax.tree.map(lambda total, value: total + value, grad_sum, grads),
                loss_sum + loss,
                saturation_sum + aux.get("tanh_sat_frac", 0.0),
            )

        zero_grads = jax.tree.map(jnp.zeros_like, state.params)
        grad_sum, loss_sum, saturation_sum = jax.lax.fori_loop(
            0, count, body, (zero_grads, jnp.asarray(0.0), jnp.asarray(0.0))
        )
        mean_grads = jax.tree.map(lambda value: value / count, grad_sum)
        metrics = {
            "loss": loss_sum / count,
            "grad_norm": optax.global_norm(mean_grads),
            "unroll_steps": jnp.asarray(targets.shape[1], dtype=jnp.float32),
        }
        if capture_saturation:
            metrics["tanh_sat_frac"] = saturation_sum / count
        return state.apply_gradients(grads=mean_grads), metrics

    return train_step


@jax.jit
def train_step(state: TrainState, context, targets):
    """Legacy zero-noise path; preserves CR0/CR1-v2 training exactly."""

    def loss_fn(params):
        return residual_unroll_loss(state.apply_fn, params, context, targets)

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), {
        **metrics,
        "grad_norm": optax.global_norm(grads),
    }


def main() -> None:
    args = tyro.cli(ContinuousResidualConfig)
    if args.context not in (1, 2):
        raise ValueError("Continuous residual dynamics supports context=1 or context=2")
    if args.unroll_steps not in (1, 4, 8):
        raise ValueError("Use --unroll-steps 1, 4, or 8")
    if args.context == 1 and args.unroll_steps != 1:
        raise ValueError(
            "context=1 is frozen as a CR0-only control; recurrent unrolling requires context=2"
        )
    if args.residual_scales and len(args.residual_scales) != 3:
        raise ValueError("residual_scales must be empty or provide u/v/density bounds")
    if args.context_noise_sigmas and len(args.context_noise_sigmas) != 3:
        raise ValueError(
            "context_noise_sigmas must be empty or provide u/v/density sigmas"
        )
    if args.context_noise_sigmas and args.unroll_steps != 4:
        raise ValueError(
            "context noise is frozen for the isolated CR1-v3-noise k=4 ablation only"
        )
    if args.microbatch and (args.microbatch <= 0 or args.batch % args.microbatch):
        raise ValueError("microbatch must be a positive divisor of the effective batch")
    if min(args.energy_weight, args.sobolev_weight, args.divergence_weight) < 0:
        raise ValueError("Physics weights must be nonnegative")
    if args.physics_dx == 0:
        paths = sorted(
            p
            for p in Path(args.data).glob("*.npy")
            if not p.name.endswith(".actions.npy")
        )
        if not paths:
            raise ValueError("No episodes available for physical grid spacing")
        first = next(p for p in paths if not p.name.endswith(".actions.npy"))
        args.physics_dx = 1.0 / np.load(first, mmap_mode="r").shape[2]
    if (
        any((args.energy_weight, args.sobolev_weight, args.divergence_weight))
        and not args.stats
    ):
        raise FileNotFoundError("Physics losses require training statistics")
    physics = PhysicsLossConfig(
        energy_weight=args.energy_weight,
        sobolev_weight=args.sobolev_weight,
        divergence_weight=args.divergence_weight,
        dx=args.physics_dx,
        energy_floor=args.energy_floor,
        gradient_ref=args.gradient_ref,
        periodic=args.periodic_derivatives,
    )
    if (
        sum(
            weight != 0.0
            for weight in (
                args.energy_weight,
                args.sobolev_weight,
                args.divergence_weight,
            )
        )
        > 1
    ):
        raise ValueError("physics screen runs exactly one auxiliary loss at a time")
    if physics.enabled and args.microbatch:
        raise ValueError(
            "Physics-loss microbatch accumulation is not supported; set --microbatch 0"
        )
    if physics.enabled and (args.context_noise_sigmas or args.unroll_steps != 4):
        raise ValueError("P1--P3 are frozen as clean context=2 k=4 CR1-v2 ablations")
    if physics.enabled and (
        (args.energy_weight > 0 and args.energy_floor <= 0.0)
        or (
            (args.sobolev_weight > 0 or args.divergence_weight > 0)
            and args.gradient_ref <= 0.0
        )
        or args.physics_dx <= 0.0
    ):
        raise ValueError(
            "enabled physics losses require positive dx, energy_floor, and gradient_ref"
        )
    loader = create_grain_dataloader(
        args.data,
        batch_size=args.batch,
        context=args.context,
        pred=args.unroll_steps,
        seed=args.seed,
        num_workers=args.grain_workers,
        stats_path=args.stats or None,
    )
    data_iter = iter(loader)
    context0, _ = next(data_iter)
    _, _, h, w, channels = context0.shape
    model = ContinuousResidualUNet(
        channels=channels,
        context_frames=args.context,
        hidden_dims=args.hidden_dims,
        num_res_blocks=args.num_res_blocks,
        residual_scales=args.residual_scales,
        stable=args.stable,
    )
    params = model.init(
        jax.random.PRNGKey(args.seed), jnp.asarray(context0), training=True
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
        if args.stable
        else optax.adam(args.lr),
    )
    out = prepare_run(
        args.out,
        (
            "continuous_residual_cr0"
            if args.unroll_steps == 1
            else "continuous_residual_cr1_physics"
            if physics.enabled
            else "continuous_residual_cr1_v3_noise"
            if args.context_noise_sigmas
            else "continuous_residual_cr1_v3_k8"
            if args.unroll_steps == 8
            else "continuous_residual_cr1_v2_k4_bounded"
            if args.residual_scales
            else "continuous_residual_cr1_k4"
        ),
        args,
        resume=args.resume,
    )
    if args.stats:
        import shutil

        destination = out / "stats.npz"
        if Path(args.stats).resolve() != destination.resolve():
            shutil.copy2(args.stats, destination)
    logger = TrainingLogger(out, "continuous_residual", args.log_every, bool(args.tb))
    noise_key = jax.random.PRNGKey(args.seed + 1)
    noisy_train_step = (
        make_train_step(
            args.context_noise_sigmas, capture_saturation=bool(args.residual_scales)
        )
        if (args.context_noise_sigmas or args.stable or args.residual_scales)
        else None
    )
    microbatch_train_step = (
        make_microbatch_train_step(
            args.microbatch,
            args.context_noise_sigmas,
            capture_saturation=bool(args.residual_scales),
        )
        if args.microbatch
        else None
    )
    physics_train_step = None
    if physics.enabled:
        raw_stats = np.load(args.stats)
        if "min" in raw_stats and "max" in raw_stats:
            minimum = jnp.asarray(raw_stats["min"], dtype=jnp.float32)
            span = jnp.asarray(
                raw_stats["max"] - raw_stats["min"] + 1e-6, dtype=jnp.float32
            )
        else:
            mean = jnp.asarray(raw_stats["mean"], dtype=jnp.float32)
            std = jnp.asarray(raw_stats["std"] + 1e-6, dtype=jnp.float32)
            minimum, span = mean - std, 2 * std
        physics_train_step = make_physics_train_step(physics, minimum, span)
    if args.resume:
        state, noise_key = restore_training_state(out, state, noise_key, data_iter)
    try:
        for step in trange(int(state.step), args.steps):
            context, targets = next(data_iter)
            if physics_train_step is not None:
                state, metrics = physics_train_step(
                    state, jnp.asarray(context), jnp.asarray(targets)
                )
            elif microbatch_train_step is not None:
                noise_key, step_key = jax.random.split(noise_key)
                state, metrics = microbatch_train_step(
                    state, jnp.asarray(context), jnp.asarray(targets), step_key
                )
            elif noisy_train_step is None:
                state, metrics = train_step(
                    state, jnp.asarray(context), jnp.asarray(targets)
                )
            else:
                noise_key, step_key = jax.random.split(noise_key)
                state, metrics = noisy_train_step(
                    state, jnp.asarray(context), jnp.asarray(targets), step_key
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
                save_training_state(out, state, noise_key)
    finally:
        logger.close()
    save_training_state(out, state, noise_key)
    save_params(out, "final", state.params)
    save_params(out, "latest", state.params)
    if args.unroll_steps == 1:
        kind = "continuous_residual_cr0"
    elif physics.enabled:
        kind = "continuous_residual_cr1_physics"
    elif args.context_noise_sigmas:
        kind = "continuous_residual_cr1_v3_noise"
    elif args.unroll_steps == 8:
        kind = "continuous_residual_cr1_v3_k8"
    elif args.residual_scales:
        kind = "continuous_residual_cr1_v2_k4_bounded"
    else:
        kind = "continuous_residual_cr1_k4"
    print(f"Saved {kind} checkpoint to {out}")


if __name__ == "__main__":
    main()
