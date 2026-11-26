"""Validate episode splits by seeds and exact array content, never by filenames."""

from pathlib import Path
import hashlib
import json


def episode_manifest(directory):
    if not Path(directory).is_dir():
        raise FileNotFoundError("Dataset split directory does not exist")
    entries = []
    for path in sorted(Path(directory).glob("*.npy")):
        if path.name.endswith(".actions.npy"):
            continue
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        entries.append(
            {
                "file": path.name,
                "seed": meta.get("seed"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not entries:
        raise ValueError("Dataset split has no episodes")
    return entries


def assert_disjoint(train_dir, eval_dir):
    return assert_disjoint_manifests(
        episode_manifest(train_dir), episode_manifest(eval_dir)
    )


def assert_disjoint_manifests(train, evaluation):
    if {x["sha256"] for x in train} & {x["sha256"] for x in evaluation}:
        raise ValueError("Train/evaluation content overlap")
    if any(x["seed"] is None for x in train + evaluation):
        raise ValueError("Seed metadata is required to certify a disjoint split")
    if {x["seed"] for x in train} & {x["seed"] for x in evaluation}:
        raise ValueError("Train/evaluation seed overlap")
    return {"train": train, "heldout": evaluation, "disjoint": True}


def assert_checkpoint_evaluation(saved_config, eval_dir):
    """Repaired models must prove separation; legacy results remain auditable."""
    if (
        saved_config.get("stable")
        or saved_config.get("modern")
        or (saved_config.get("architecture_version") or 0) >= 2
    ):
        if saved_config.get("training_manifest"):
            return assert_disjoint_manifests(
                saved_config["training_manifest"], episode_manifest(eval_dir)
            )
        return assert_disjoint(saved_config["data"], eval_dir)
    return {
        "disjoint": False,
        "status": "legacy protocol; no generalization certification",
    }
