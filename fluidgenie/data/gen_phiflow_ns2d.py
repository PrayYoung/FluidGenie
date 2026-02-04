"""
PhiFlow NS2D data generator (JAX backend) — stable version

- Avoids Domain (uses Box)
- Uses explicit diffusion by default (more compatible across PhiFlow versions)
- Uses smaller dt by default
- Exports numpy arrays with stable dimension order
"""

from __future__ import annotations

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
    dt: float = 0.1

    resolution: int = 128
    viscosity: float = 0.001
    use_implicit_diffusion: bool = False

    with_density: bool = True

    # Optional internal substeps per saved frame
    substeps: int = 1

    # For future forcing (disabled by default)
    forcing_strength: float = 0.0
    forcing_radius: float = 0.08


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def init_state(cfg: NS2DConfig):
    bounds = Box(x=(0, 1), y=(0, 1))

    velocity = StaggeredGrid(
        0.0,
        bounds=bounds,
        x=cfg.resolution,
        y=cfg.resolution,
    )

    density = None
    if cfg.with_density:
        density = CenteredGrid(
            0.0,
            bounds=bounds,
            x=cfg.resolution,
            y=cfg.resolution,
        )
        r = _rng(cfg.seed)
        cx, cy = r.uniform(0.3, 0.7, size=2)
        blob = Sphere(x=float(cx), y=float(cy), radius=float(cfg.forcing_radius))
        density = density + 1.0 * resample(blob, density)

    return velocity, density


def step_state(cfg: NS2DConfig, velocity, density):
    # Advection
    velocity = advect.semi_lagrangian(velocity, velocity, cfg.dt)
    if density is not None:
        density = advect.semi_lagrangian(density, velocity, cfg.dt)

    # Diffusion: explicit is more compatible across PhiFlow versions / backends
    if cfg.viscosity > 0:
        if cfg.use_implicit_diffusion:
            velocity = diffuse.implicit(velocity, cfg.viscosity, cfg.dt)
        else:
            velocity = diffuse.explicit(velocity, cfg.viscosity, cfg.dt)

    # Projection to make velocity divergence-free
    velocity, _ = fluid.make_incompressible(velocity)

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
    velocity, density = init_state(cfg)

    frames = []
    for _t in range(cfg.steps):
        frames.append(state_to_numpy(velocity, density))

        # advance substeps per saved frame
        for _ in range(max(1, cfg.substeps)):
            velocity, density = step_state(cfg, velocity, density)

    fields = np.stack(frames, axis=0)
    meta = asdict(cfg)
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


if __name__ == "__main__":
    cfg = NS2DConfig(seed=0, steps=50, resolution=64, dt=0.1, with_density=True, substeps=1)
    generate_dataset("data/ns2d_test", episodes=2, cfg=cfg)
    print("Wrote data/ns2d_test/*.npz")
