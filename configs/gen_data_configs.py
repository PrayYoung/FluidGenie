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
    with_density: bool = True
    substeps: int = 5
    save_every: int = 1
    jit_step: bool = True
    forcing_strength: float = 0.0
    forcing_radius: float = 0.08
    density_radius: float = 0.08
    pressure_solver: str = "CG"
    pressure_rel_tol: float = 1e-4
    pressure_abs_tol: float = 1e-4
    pressure_max_iters: int = 200


@dataclass
class GenDataArgs:
    out: str = "data/ns2d"
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
