from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NS2DConfig:
    seed: int = 0
    steps: int = 200
    dt: float = 0.02
    resolution: int = 128
    viscosity: float = 0.001
    use_implicit_diffusion: bool = False
    init_velocity_noise: float = 0.10
    init_velocity_scale: float = 0.3
    init_velocity_smoothness: float = 1.5
    with_density: bool = True
    substeps: int = 5
    save_every: int = 1
    jit_step: bool = True
    forcing_strength: float = 0.02
    forcing_radius: float = 0.3
    forcing_smoothness: float = 2.0
    linear_drag: float = 0.1
    density_decay: float = 0.05
    density_source_strength: float = 0.05
    density_radius: float = 0.08
    pressure_solver: str = "CG"
    pressure_rel_tol: float = 1e-4
    pressure_abs_tol: float = 1e-4
    pressure_max_iters: int = 200
    burn_in: int = 1000
    action_mode: str = "none"
    action_strength: float = 0.05
    action_hold: int = 8
    action_hold_min: int = 2
    action_hold_max: int = 4
    action_radius: float = 0.14
    held_action_id: int = -1
    action_start: int = 0


@dataclass
class GenDataArgs:
    out: str = "data/raw/ns2d"
    episodes: int = 200
    prefix: str = "episode"
    steps: int = 120
    res: int = 96
    dt: float = 0.02
    viscosity: float = 0.001
    implicit: bool = False
    density: int = 1
    substeps: int = 1
    save_every: int = 1
    seed: int = 0
    noise: float = 0.05
    forcing_strength: float = 0.005
    forcing_radius: float = 0.10
    density_radius: float = 0.08
    burn_in: int = 1000
    action_mode: str = "none"
    action_strength: float = 0.05
    action_hold: int = 8
    action_hold_min: int = 2
    action_hold_max: int = 4
    action_radius: float = 0.14

    init_velocity_scale: float = 0.3
    init_velocity_smoothness: float = 1.5
    forcing_smoothness: float = 2.0
    linear_drag: float = 0.1
    density_decay: float = 0.05
    density_source_strength: float = 0.05
    pressure_solver: str = "CG"
    pressure_rel_tol: float = 1e-4
    pressure_abs_tol: float = 1e-4
    pressure_max_iters: int = 200
    jit_step: bool = True
    held_action_id: int = -1
    action_start: int = 0


def forced_control_config(**overrides):
    """Versioned unsteady-flow preset; legacy NS2D defaults remain reproducible."""
    values = dict(
        forcing_strength=1.0,
        init_velocity_noise=1.0,
        viscosity=0.0005,
        linear_drag=0.1,
        use_implicit_diffusion=True,
        dt=0.02,
        substeps=5,
        forcing_radius=0.3,
        forcing_smoothness=2.0,
    )
    values.update(overrides)
    return NS2DConfig(**values)
