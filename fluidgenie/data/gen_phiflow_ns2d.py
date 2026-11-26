"""
PhiFlow NS2D data generator (JAX backend) — stable version

- Avoids Domain (uses Box)
- Uses explicit diffusion by default (more compatible across PhiFlow versions)
- Uses smaller dt by default
- Exports numpy arrays with stable dimension order
- Adds optional JIT stepping and configurable frame stride
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict, fields
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json

from jaxtyping import Array, Float

import numpy as np
import tyro
from tqdm import tqdm

from phi.jax.flow import *  # noqa: F401,F403
from configs.gen_data_configs import NS2DConfig, GenDataArgs


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed) & 0xFFFFFFFF)


def init_state(cfg: NS2DConfig):
    bounds = Box(x=(0, 1), y=(0, 1))
    math.seed(int(cfg.seed) & 0xFFFFFFFF)

    velocity = StaggeredGrid(
        Noise(
            math.channel(vector="x,y"),
            scale=cfg.init_velocity_scale,
            smoothness=cfg.init_velocity_smoothness,
        ),
        extrapolation=extrapolation.PERIODIC,
        bounds=bounds,
        x=cfg.resolution,
        y=cfg.resolution,
    )

    velocity = velocity * cfg.init_velocity_noise

    density = None
    forcing = None
    if cfg.with_density:
        density = CenteredGrid(
            0.0,
            extrapolation=extrapolation.PERIODIC,
            bounds=bounds,
            x=cfg.resolution,
            y=cfg.resolution,
        )
        r = _rng(cfg.seed)
        cx, cy = (
            (0.5, 0.5)
            if cfg.action_mode == "joystick5_localized"
            else r.uniform(0.3, 0.7, size=2)
        )
        blob = Sphere(x=float(cx), y=float(cy), radius=float(cfg.density_radius))
        density = density + 1.0 * resample(blob, density)
    else:
        r = _rng(cfg.seed)
        cx, cy = (
            (0.5, 0.5)
            if cfg.action_mode == "joystick5_localized"
            else r.uniform(0.3, 0.7, size=2)
        )

    if cfg.forcing_strength > 0:
        forcing = CenteredGrid(
            Noise(
                math.channel(vector="x,y"),
                scale=cfg.forcing_radius,
                smoothness=cfg.forcing_smoothness,
            ),
            extrapolation=extrapolation.PERIODIC,
            bounds=bounds,
            x=cfg.resolution,
            y=cfg.resolution,
        )

    if forcing is not None:
        forcing = forcing * cfg.forcing_strength
    return velocity, density, forcing


def _pressure_solve(cfg: NS2DConfig, pressure=None):
    return Solve(
        x0=pressure,
        method=cfg.pressure_solver,
        rel_tol=cfg.pressure_rel_tol,
        abs_tol=cfg.pressure_abs_tol,
        max_iterations=cfg.pressure_max_iters,
    )


def step_state(
    cfg: NS2DConfig, velocity, density, forcing, pressure_solve, density_source=None
):
    # Advection
    velocity = advect.semi_lagrangian(velocity, velocity, cfg.dt)
    if density is not None:
        density = advect.semi_lagrangian(density, velocity, cfg.dt)
        density = density * float(np.exp(-cfg.density_decay * cfg.dt))
        if density_source is not None:
            density = density + density_source * (cfg.density_source_strength * cfg.dt)

    # External body force (vortex-like), if enabled
    if forcing is not None:
        velocity = velocity + resample(forcing, velocity) * cfg.dt

    velocity = velocity * float(np.exp(-cfg.linear_drag * cfg.dt))

    # Diffusion: explicit is more compatible across PhiFlow versions / backends
    if cfg.viscosity > 0:
        if cfg.use_implicit_diffusion:
            velocity = diffuse.implicit(
                velocity,
                cfg.viscosity,
                cfg.dt,
                solve=Solve(
                    method=cfg.pressure_solver,
                    rel_tol=cfg.pressure_rel_tol,
                    abs_tol=cfg.pressure_abs_tol,
                    max_iterations=cfg.pressure_max_iters,
                ),
            )
        else:
            velocity = diffuse.explicit(velocity, cfg.viscosity, cfg.dt)

    # Projection to make velocity divergence-free
    velocity, pressure = fluid.make_incompressible(velocity, solve=pressure_solve)

    return velocity, density, pressure


def _to_numpy_stable(phi_tensor, order: Optional[Tuple[str, ...]] = None) -> np.ndarray:
    """
    Convert Φ-ML tensor to numpy across PhiFlow/phiml versions.

    We try a few conversion routes in order of preference:
    1) phiml math.reshaped_numpy() with explicit dim order
    2) phiml math.numpy() (backend-aware)
    3) tensor.numpy() if available
    4) np.asarray fallback
    """
    # 1) Preferred: explicit dimension order to avoid warnings / ambiguity
    if order is not None:
        # Newer PhiML: Tensor.numpy(order=...)
        try:
            return np.asarray(phi_tensor.numpy(order=order))
        except Exception:
            pass
        # Older PhiML: math.reshaped_numpy(value, groups)
        try:
            return np.asarray(math.reshaped_numpy(phi_tensor, order))
        except Exception:
            pass

    # 2) backend conversion (may warn about dimension order)
    try:
        arr = math.numpy(phi_tensor)
        return np.asarray(arr)
    except Exception:
        pass

    # 3) Some versions expose .numpy()
    try:
        return np.asarray(phi_tensor.numpy())
    except Exception:
        pass

    # 4) Fallback
    return np.asarray(phi_tensor)


def state_to_numpy(velocity, density) -> Float[Array, "h w c"]:
    # centered velocity samples: [y, x, vector]
    v_center = velocity.at_centers().values
    v_np = _to_numpy_stable(v_center, order=("y", "x", "vector")).astype(np.float32)

    if density is None:
        return v_np

    d_np = _to_numpy_stable(density.values, order=("y", "x")).astype(np.float32)[
        ..., None
    ]
    return np.concatenate([v_np, d_np], axis=-1)


def localized_jet_kernel(resolution, radius):
    coordinates = (np.arange(resolution) + 0.5) / resolution
    distance = np.minimum(np.abs(coordinates - 0.5), 1 - np.abs(coordinates - 0.5))
    kernel = np.exp(
        -(distance[:, None] ** 2 + distance[None, :] ** 2) / (2 * radius * radius)
    )
    return (kernel - kernel.mean()).astype(np.float32)


_PHYSICS_FIELDS = (
    "dt",
    "viscosity",
    "use_implicit_diffusion",
    "linear_drag",
    "density_decay",
    "density_source_strength",
    "pressure_solver",
    "pressure_rel_tol",
    "pressure_abs_tol",
    "pressure_max_iters",
)


@lru_cache(maxsize=16)
def _cached_stepper(physics, use_jit, has_force=True, with_density=True):
    cfg = NS2DConfig(**dict(zip(_PHYSICS_FIELDS, physics)))

    def advance(velocity, density, pressure, force, source):
        return step_state(
            cfg, velocity, density, force, _pressure_solve(cfg, pressure), source
        )

    return math.jit_compile(advance) if use_jit else advance


def _initialize_simulation(cfg):
    """Create and burn in one immutable simulator state and its controls."""
    velocity, density, forcing = init_state(cfg)
    velocity, pressure = fluid.make_incompressible(velocity, solve=_pressure_solve(cfg))

    controls = None
    if cfg.action_mode.startswith("joystick5"):
        if forcing is None:
            forcing = CenteredGrid(
                math.vec(x=0.0, y=0.0),
                extrapolation=extrapolation.PERIODIC,
                bounds=velocity.bounds,
                x=cfg.resolution,
                y=cfg.resolution,
            )
        vectors = ((0.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0))
        kernel = (
            math.tensor(
                localized_jet_kernel(cfg.resolution, cfg.action_radius),
                math.spatial(y=cfg.resolution, x=cfg.resolution),
            )
            if cfg.action_mode == "joystick5_localized"
            else 1.0
        )
        controls = [
            CenteredGrid(
                kernel * math.vec(x=x, y=y),
                extrapolation=extrapolation.PERIODIC,
                bounds=velocity.bounds,
                x=cfg.resolution,
                y=cfg.resolution,
            )
            * cfg.action_strength
            for x, y in vectors
        ]
    density_source = density

    step_fn = _cached_stepper(
        tuple(getattr(cfg, name) for name in _PHYSICS_FIELDS),
        cfg.jit_step,
        forcing is not None,
        density is not None,
    )
    for _ in range(cfg.burn_in):
        velocity, density, pressure = step_fn(
            velocity, density, pressure, forcing, density_source
        )
    return velocity, density, pressure, forcing, controls, density_source, step_fn


def generate_episode(
    cfg: NS2DConfig, fixed_actions=None
) -> Tuple[Float[Array, "t h w c"], Dict[str, Any]]:
    if (
        cfg.action_mode not in {"none", "joystick5", "joystick5_localized"}
        or cfg.action_hold < 1
    ):
        raise ValueError(
            "action_mode must be none or joystick5, with positive action_hold"
        )
    if cfg.steps < 1 or cfg.burn_in < 0 or cfg.substeps < 1 or cfg.save_every < 1:
        raise ValueError("Invalid episode length, burn-in, substeps, or save stride")
    if cfg.action_mode.startswith("joystick5") and cfg.save_every != 1:
        raise ValueError(
            "Action-conditioned data requires save_every=1 for exact transition labels"
        )
    if cfg.held_action_id not in range(-1, 5):
        raise ValueError("held_action_id must be -1 or a joystick5 action")
    if fixed_actions is not None:
        fixed_actions = np.asarray(fixed_actions, dtype=np.int32)
        if (
            fixed_actions.shape != (cfg.steps,)
            or np.any(fixed_actions < 0)
            or np.any(fixed_actions > 4)
        ):
            raise ValueError(
                "fixed_actions must provide one joystick5 ID per stored simulation step"
            )
    if (
        cfg.action_hold_min < 1
        or cfg.action_hold_max < cfg.action_hold_min
        or cfg.action_radius <= 0
    ):
        raise ValueError("Invalid joystick hold range or action radius")
    velocity, density, pressure, forcing, controls, density_source, step_fn = (
        _initialize_simulation(cfg)
    )
    frames, actions = [], []
    rng = _rng(
        (int(cfg.seed) * 100003 + 17) & 0xFFFFFFFF
        if cfg.action_mode == "joystick5_localized"
        else cfg.seed + 1000000
    )
    action = 0
    next_switch = 0
    for t in range(cfg.steps):
        if cfg.action_mode == "joystick5_localized" and t >= next_switch:
            choices = [i for i in range(5) if i != action]
            action = int(rng.choice(choices))
            next_switch = t + int(
                rng.randint(cfg.action_hold_min, cfg.action_hold_max + 1)
            )
        elif t % cfg.action_hold == 0 and cfg.action_mode == "joystick5":
            action = (
                cfg.held_action_id if cfg.held_action_id >= 0 else int(rng.randint(5))
            )
        if cfg.held_action_id >= 0 and cfg.action_mode.startswith("joystick5"):
            action = cfg.held_action_id if t >= cfg.action_start else 0
        if fixed_actions is not None:
            action = int(fixed_actions[t])
        if t % cfg.save_every == 0:
            frames.append(state_to_numpy(velocity, density))
            actions.append(action)
        body_force = forcing if controls is None else forcing + controls[action]
        for _ in range(cfg.substeps):
            velocity, density, pressure = step_fn(
                velocity, density, pressure, body_force, density_source
            )
    fields = np.stack(frames, axis=0)
    meta = asdict(cfg)
    meta.update(
        saved_frames=int(fields.shape[0]),
        boundary="periodic",
        actions=actions,
        action_names=["noop", "up", "down", "right", "left"],
        protocol="forced_periodic_v3",
    )
    return fields, meta


def generate_counterfactual_branches(cfg: NS2DConfig, actions=range(5)):
    """Return {action: (fields, metadata)} forked from one exact shared state.

    Frames are recorded before each action. Consequently, action_start=1
    shares frames 0 and 1; action 1 first affects frame 2. PhiFlow fields and
    JAX arrays are immutable: each branch advances its own returned state.
    """
    actions = tuple(actions)
    if (
        not actions
        or len(set(actions)) != len(actions)
        or any(a not in range(5) for a in actions)
    ):
        raise ValueError("Provide distinct joystick action IDs in 0..4")
    if not cfg.action_mode.startswith("joystick5") or cfg.save_every != 1:
        raise ValueError(
            "Counterfactual branches require joystick controls and save_every=1"
        )
    if not 0 <= cfg.action_start < cfg.steps - 1 or cfg.burn_in < 0 or cfg.substeps < 1:
        raise ValueError(
            "Invalid counterfactual prefix, horizon, or integration settings"
        )
    velocity, density, pressure, forcing, controls, source, advance = (
        _initialize_simulation(cfg)
    )
    shared_frames = [state_to_numpy(velocity, density)]
    for _ in range(cfg.action_start):
        for _ in range(cfg.substeps):
            velocity, density, pressure = advance(
                velocity, density, pressure, forcing + controls[0], source
            )
        shared_frames.append(state_to_numpy(velocity, density))
    fork_state = (velocity, density, pressure)
    branches = {}
    for action in actions:
        velocity, density, pressure = fork_state
        frames = list(shared_frames)
        for t in range(cfg.action_start, cfg.steps):
            if t > cfg.action_start:
                frames.append(state_to_numpy(velocity, density))
            for _ in range(cfg.substeps):
                velocity, density, pressure = advance(
                    velocity, density, pressure, forcing + controls[action], source
                )
        fields = np.stack(frames)
        metadata = asdict(cfg)
        metadata.update(
            held_action_id=action,
            saved_frames=len(fields),
            boundary="periodic",
            actions=[0] * cfg.action_start + [action] * (cfg.steps - cfg.action_start),
            action_names=["noop", "up", "down", "right", "left"],
            protocol="forced_periodic_v3",
            counterfactual_protocol="shared_state_v1",
            shared_prefix_frames=cfg.action_start + 1,
        )
        branches[action] = fields, metadata
    return branches


def save_episode_npy(
    out_path: Path, fields: Float[Array, "t h w c"], meta: Dict[str, Any]
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite episode: {out_path}")
    np.save(out_path, fields)
    if meta.get("action_mode", "none").startswith("joystick5"):
        action_dir = out_path.parent / "actions"
        action_dir.mkdir(exist_ok=True)
        np.save(action_dir / out_path.name, np.asarray(meta["actions"], dtype=np.int32))
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def generate_dataset(
    out_dir: str,
    episodes: int = 10,
    cfg: Optional[NS2DConfig] = None,
    name_prefix: str = "episode",
):
    out_dir = Path(out_dir)
    cfg = cfg or NS2DConfig()

    for i in tqdm(range(episodes), desc="episodes"):
        ep_cfg = NS2DConfig(**asdict(cfg))
        ep_cfg.seed = cfg.seed + i
        target = out_dir / f"{name_prefix}_{i:06d}.npy"
        if target.exists():
            metadata = target.with_suffix(".json")
            previous = json.loads(metadata.read_text()) if metadata.exists() else {}
            if any(previous.get(k) != v for k, v in asdict(ep_cfg).items()):
                raise ValueError(
                    "Existing episode metadata does not match the requested configuration"
                )
            np.load(
                target, mmap_mode="r"
            )  # Reject truncated output rather than silently skipping it.
            if ep_cfg.action_mode.startswith("joystick5"):
                action_path = target.parent / "actions" / target.name
                if not action_path.exists() or not np.array_equal(
                    np.load(action_path),
                    np.asarray(previous.get("actions", []), dtype=np.int32),
                ):
                    raise ValueError(
                        "Existing episode is missing its matching action sidecar"
                    )
            continue
        generated, meta = generate_episode(ep_cfg)
        save_episode_npy(target, generated, meta)


def config_from_args(args):
    aliases = {
        "resolution": "res",
        "init_velocity_noise": "noise",
        "use_implicit_diffusion": "implicit",
        "with_density": "density",
    }
    values = {
        field.name: getattr(args, aliases.get(field.name, field.name))
        for field in fields(NS2DConfig)
    }
    values["with_density"] = bool(values["with_density"])
    return NS2DConfig(**values)


def main():
    args = tyro.cli(GenDataArgs)
    cfg = config_from_args(args)
    generate_dataset(args.out, episodes=args.episodes, cfg=cfg, name_prefix=args.prefix)
    print(f"Wrote {args.out}/*.npy")


if __name__ == "__main__":
    main()
