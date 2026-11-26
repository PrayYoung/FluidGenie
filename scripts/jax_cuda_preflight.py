"""Fail-fast CUDA/JAX gate for the exact remote workload environment."""

from __future__ import annotations

import json
import subprocess
import jax
import jax.numpy as jnp
import jaxlib
from jax import lax


def driver_diagnostics():
    """Capture a small, nonsecret host-driver summary before JAX initialization."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {
            "return_code": result.returncode,
            "gpu_driver_memory": result.stdout.strip(),
            "error": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"return_code": None, "error": type(error).__name__}


def main() -> None:
    print("driver_diagnostics=", json.dumps(driver_diagnostics()), flush=True)
    print("jax=", jax.__version__, "jaxlib=", jaxlib.__version__)
    devices = jax.devices()
    print(
        "devices=",
        [
            {
                "platform": d.platform,
                "kind": getattr(d, "device_kind", "unknown"),
                "id": d.id,
            }
            for d in devices
        ],
        flush=True,
    )
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("JAX CUDA GPU is required before CR1-v2 training")
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 8, 8, 3), dtype=jnp.float32)
    math = jnp.sum(x @ jnp.ones((3, 3), dtype=jnp.float32)).block_until_ready()
    kernel = jnp.ones((3, 3, 3, 4), dtype=jnp.float32)
    conv = lax.conv_general_dilated(
        x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    conv.block_until_ready()
    print("preflight_math=", float(math), "conv_shape=", conv.shape)


if __name__ == "__main__":
    main()
