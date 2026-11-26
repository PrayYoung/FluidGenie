"""Driver diagnostics remain useful even when the CUDA host is unavailable."""

import subprocess
from types import SimpleNamespace
import pytest
from scripts.jax_cuda_preflight import driver_diagnostics


def test_driver_summary_records_device_without_secrets(monkeypatch):
    def completed(command, **kwargs):
        assert command[0] == "nvidia-smi" and kwargs["timeout"] == 20
        return SimpleNamespace(
            returncode=0, stdout="NVIDIA GPU, 570.0, 16384 MiB\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", completed)
    assert driver_diagnostics() == {
        "return_code": 0,
        "gpu_driver_memory": "NVIDIA GPU, 570.0, 16384 MiB",
        "error": "",
    }


@pytest.mark.parametrize(
    "error", [FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 20)]
)
def test_driver_summary_survives_missing_or_unresponsive_host(monkeypatch, error):
    def failed(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", failed)
    assert driver_diagnostics()["error"] == type(error).__name__
