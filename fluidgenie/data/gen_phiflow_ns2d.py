"""
PhiFlow NS2D data generator (JAX backend) — stable version

- Avoids Domain (uses Box)
- Uses explicit diffusion by default (more compatible across PhiFlow versions)
- Uses smaller dt by default
- Exports numpy arrays with stable dimension order
- Adds optional JIT stepping and configurable frame stride
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

from phi.jax.flow import *  # noqa: F401,F403


@dataclass
class NS2DConfig:
    seed: int = 0
    steps: int = 200

    # Use a smaller dt by default to avoid CFL issues
    dt: float = 0.02

    resolution: int = 128
    viscosity: float = 0.001
    use_implicit_diffusion: bool = False
    init_velocity_noise: float = 0.10

    with_density: bool = True

    # Optional internal substeps per saved frame
    substeps: int = 5
    save_every: int = 1
    jit_step: bool = True

    # External forcing and initialization
    forcing_strength: float = 0.0
    forcing_radius: float = 0.08
    density_radius: float = 0.08

    # Pressure solve tuning (main runtime hotspot)
    pressure_solver: str = "CG"
    pressure_rel_tol: float = 1e-4
    pressure_abs_tol: float = 1e-4
    pressure_max_iters: int = 200


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def init_state(cfg: NS2DConfig):
    bounds = Box(x=(0, 1), y=(0, 1))
    math.seed(cfg.seed)

    velocity = StaggeredGrid(
        Noise(scale=cfg.init_velocity_noise),
        bounds=bounds,
        x=cfg.resolution,
        y=cfg.resolution,
    )

    density = None
    forcing = None
    if cfg.with_density:
        density = CenteredGrid(
            0.0,
            bounds=bounds,
            x=cfg.resolution,
            y=cfg.resolution,
        )
        r = _rng(cfg.seed)
        cx, cy = r.uniform(0.3, 0.7, size=2)
        blob = Sphere(x=float(cx), y=float(cy), radius=float(cfg.density_radius))
        density = density + 1.0 * resample(blob, density)
    else:
        r = _rng(cfg.seed)
        cx, cy = r.uniform(0.3, 0.7, size=2)

    if cfg.forcing_strength > 0:
        forcing = CenteredGrid(
            Noise(scale=cfg.forcing_strength, smoothness=max(1.0, cfg.forcing_radius * 30.0)),
            bounds=bounds,
            x=cfg.resolution,
            y=cfg.resolution,
        )

    return velocity, density, forcing


def _pressure_solve(cfg: NS2DConfig):
    return Solve(
        method=cfg.pressure_solver,
        rel_tol=cfg.pressure_rel_tol,
        abs_tol=cfg.pressure_abs_tol,
        max_iterations=cfg.pressure_max_iters,
    )


def step_state(cfg: NS2DConfig, velocity, density, forcing, pressure_solve):
    # Advection
    velocity = advect.semi_lagrangian(velocity, velocity, cfg.dt)
    if density is not None:
        density = advect.semi_lagrangian(density, velocity, cfg.dt)

    # External body force (vortex-like), if enabled
    if forcing is not None:
        velocity = velocity + resample(forcing, velocity) * cfg.dt

    # Diffusion: explicit is more compatible across PhiFlow versions / backends
    if cfg.viscosity > 0:
        if cfg.use_implicit_diffusion:
            velocity = diffuse.implicit(velocity, cfg.viscosity, cfg.dt)
        else:
            velocity = diffuse.explicit(velocity, cfg.viscosity, cfg.dt)

    # Projection to make velocity divergence-free
    velocity, _ = fluid.make_incompressible(velocity, solve=pressure_solve)

    return velocity, density


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


def state_to_numpy(velocity, density) -> np.ndarray:
    # centered velocity samples: [y, x, vector]
    v_center = velocity.at_centers().values
    v_np = _to_numpy_stable(v_center, order=("y", "x", "vector")).astype(np.float32)

    if density is None:
        return v_np

    d_np = _to_numpy_stable(density.values, order=("y", "x")).astype(np.float32)[..., None]
    return np.concatenate([v_np, d_np], axis=-1)


def generate_episode(cfg: NS2DConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    velocity, density, forcing = init_state(cfg)
    pressure_solve = _pressure_solve(cfg)
    save_every = max(1, int(cfg.save_every))
    velocity, _ = fluid.make_incompressible(velocity, solve=pressure_solve)

    if cfg.with_density:
        def _step_fn(v, d):
            return step_state(cfg, v, d, forcing, pressure_solve)

        step_fn = math.jit_compile(_step_fn) if cfg.jit_step else _step_fn
    else:
        def _step_v_only(v):
            v, _ = step_state(cfg, v, None, forcing, pressure_solve)
            return v

        step_v_only = math.jit_compile(_step_v_only) if cfg.jit_step else _step_v_only

    frames = []
    for _t in range(cfg.steps):
        if _t % save_every == 0:
            frames.append(state_to_numpy(velocity, density))

        # advance substeps per saved frame
        for _ in range(max(1, cfg.substeps)):
            if cfg.with_density:
                velocity, density = step_fn(velocity, density)
            else:
                velocity = step_v_only(velocity)

    fields = np.stack(frames, axis=0)
    meta = asdict(cfg)
    meta["saved_frames"] = int(fields.shape[0])
    return fields, meta


def save_episode_npz(out_path: Path, fields: np.ndarray, meta: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, fields=fields, meta=meta)


def generate_dataset(out_dir: str, episodes: int = 10, cfg: Optional[NS2DConfig] = None, name_prefix: str = "episode"):
    out_dir = Path(out_dir)
    cfg = cfg or NS2DConfig()

    for i in range(episodes):
        ep_cfg = NS2DConfig(**asdict(cfg))
        ep_cfg.seed = cfg.seed + i
        fields, meta = generate_episode(ep_cfg)
        save_episode_npz(out_dir / f"{name_prefix}_{i:06d}.npz", fields, meta)
