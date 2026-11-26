"""Keep fresh experiments from silently mixing logs or overwriting checkpoints."""

from pathlib import Path
from dataclasses import asdict
from configs.config_io import save_config_json


def prepare_run(directory, kind, config, resume=False):
    directory = Path(directory)
    if resume:
        import json

        if not (directory / "train_state").exists():
            raise FileNotFoundError(
                "No resumable training state exists in the requested run"
            )
        saved = json.loads((directory / "config.json").read_text())
        current = asdict(config)
        ignored = {
            "resume",
            "log_every",
            "tb",
            "checkpoint_every",
            "training_manifest",
            "grain_workers",
        }
        if saved.get("kind") != kind:
            raise ValueError(
                "Resume checkpoint kind does not match the requested trainer"
            )
        # JSON canonicalization handles tuples serialized as lists.
        for key in set(saved["config"]) | set(current):
            value = saved["config"].get(key)
            if key not in ignored and json.dumps(value, sort_keys=True) != json.dumps(
                current.get(key), sort_keys=True
            ):
                raise ValueError(f"Resume configuration mismatch: {key}")
        return directory
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            "Training output is nonempty; use a fresh experiment directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    if payload.get("data"):
        from fluidgenie.data.splits import episode_manifest

        payload["training_manifest"] = episode_manifest(payload["data"])
    save_config_json(
        directory, {"kind": kind, "config": payload, "implementation_version": 2}
    )
    return directory
