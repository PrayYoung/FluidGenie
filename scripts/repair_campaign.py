"""One bounded repair campaign. No provisioning or provider credentials live here.

Each stage writes a journal entry before execution. A failed engineering or data
quality gate stops the campaign; negative scientific scores are retained.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def verify_reusable_data(data, resolution, frames, train_count, heldout_count, action):
    """Certify saved episodes against their original split manifest before reuse."""
    from fluidgenie.data.splits import assert_disjoint

    saved = json.loads((data / "split_manifest.json").read_text())
    actual = assert_disjoint(data / "train", data / "heldout")
    for split, count in [("train", train_count), ("heldout", heldout_count)]:
        if len(actual[split]) != count or actual[split] != saved[split]:
            raise ValueError(f"Saved {split} episodes differ from the certified manifest")
        for episode in actual[split]:
            array = np.load(data / split / episode["file"], mmap_mode="r")
            if array.shape != (frames, resolution, resolution, 3):
                raise ValueError("Saved episode shape differs from campaign protocol")
    config = saved["generator_config"]
    if (config["resolution"], config["steps"], config["action_mode"]) != (
        resolution, frames, action
    ) or saved.get("statistics_source") != "train":
        raise ValueError("Saved dataset protocol differs from the requested campaign")
    stats = np.load(data / "train_stats.npz")
    if not all(np.isfinite(stats[key]).all() for key in ["mean", "std"]):
        raise ValueError("Saved training statistics are nonfinite")
    if action != "none":
        for split in ["train", "heldout"]:
            for episode in actual[split]:
                labels = np.load(data / split / "actions" / episode["file"])
                meta = json.loads((data / split / episode["file"]).with_suffix(".json").read_text())
                if not np.array_equal(labels, meta["actions"]):
                    raise ValueError("Saved action labels differ from episode metadata")
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--resume-training", action="store_true")
    a = parser.parse_args()
    if a.cpu and not a.smoke:
        parser.error("--cpu is only supported for engineering smoke validation")
    campaign_name = (
        "repair_v4_cpu_smoke"
        if a.cpu
        else "repair_v4_gpu_smoke"
        if a.smoke
        else "repair_v4"
    )
    root = Path("outputs") / campaign_name
    root.mkdir(parents=True, exist_ok=True)
    journal = []
    py = sys.executable

    def run(name, args):
        if (
            a.resume_training
            and args[:1] == ["-m"]
            and args[1].startswith("fluidgenie.training.")
            and "--out" in args
        ):
            out = Path(args[args.index("--out") + 1])
            if (out / "train_state").is_dir():
                args = [*args, "--resume"]
        entry = {
            "stage": name,
            "state": "running",
            "started_unix": time.time(),
            "command": args,
        }
        journal.append(entry)
        (root / "campaign_state.json").write_text(json.dumps(journal, indent=2) + "\n")
        with (root / (name + ".log")).open("w") as log:
            result = subprocess.run([py, *args], stdout=log, stderr=subprocess.STDOUT)
        entry.update(
            state="completed" if result.returncode == 0 else "failed",
            exit_code=result.returncode,
            finished_unix=time.time(),
        )
        (root / "campaign_state.json").write_text(json.dumps(journal, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

    resolution = 16 if a.smoke else 64
    train_episodes = 3 if a.smoke else 30
    heldout_episodes = 2 if a.smoke else 10
    frames = 48 if a.smoke else 160
    cr_steps = 20 if a.smoke else 3000
    fm_steps = 30 if a.smoke else 15000
    st_steps = 20 if a.smoke else 10000
    dyn_steps = 20 if a.smoke else 15000
    spec = {
        "protocol": "periodic-disjoint-v4",
        "smoke": a.smoke,
        "reuse_data": a.reuse_data,
        "resume_training": a.resume_training,
        "resolution": resolution,
        "continuous_batch": 8 if a.smoke else 32,
        "continuous_hidden_dims": [8, 16, 32] if a.smoke else [64, 128, 256],
        "checkpoint_every": 0 if a.smoke else 1000,
        "st_model_dim": 64 if a.smoke else 256,
        "st_blocks": 2 if a.smoke else 4,
        "tokenizer_codebook": 64 if a.smoke else 256,
        "cr_steps": cr_steps,
        "fm_steps": fm_steps,
        "st_steps": st_steps,
        "lam_steps": 20 if a.smoke else 6000,
        "lam_discovery_gate_version": 4,
        "dynamics_steps": dyn_steps,
        "seeds": [0, 1, 2],
        "unseen_seed_start": 200003,
        "primary_metrics": [
            "mse",
            "motion_ratio_of_means",
            "vorticity_correlation",
            "tke",
            "enstrophy",
            "energy_spectrum",
            "persistence_mse",
        ],
        "success_criteria": "Regression/data/finite-gradient gates pass. Model quality is reported against persistence; no automatic claim of superiority. LAM semantic control requires discovery gate.",
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ["fluidgenie", "configs", "scripts"]
            for p in Path(folder).rglob("*.py")
        },
    }
    (root / "protocol.json").write_text(json.dumps(spec, indent=2) + "\n")
    dims = ["8", "16", "32"] if a.smoke else ["64", "128", "256"]
    if not a.cpu:
        run("cuda_preflight", ["scripts/jax_cuda_preflight.py"])
    run("regression_tests", ["-m", "pytest", "-q"])
    if not a.cpu:
        run(
            "bptt_cuda",
            [
                "scripts/continuous_residual_gpu_smoke.py",
                "--batch",
                "8" if a.smoke else "32",
                "--height",
                str(resolution),
                "--width",
                str(resolution),
                "--hidden-dims",
                *dims,
            ],
        )
    for name, action in [
        ("forced_control", "none"),
        ("joystick", "joystick5_localized"),
    ]:
        # "forced_control" means no joystick intervention; the periodic body forcing remains active.
        data = (Path("data") / campaign_name) / name
        if a.reuse_data:
            verify_reusable_data(
                data, resolution, frames, train_episodes, heldout_episodes, action
            )
        else:
            run(
                "generate_" + name,
                [
                    "scripts/generate_disjoint_data.py",
                    "--out",
                    str(data),
                    "--resolution",
                    str(resolution),
                    "--steps",
                    str(frames),
                    "--train-episodes",
                    str(train_episodes),
                    "--heldout-episodes",
                    str(heldout_episodes),
                    "--action-mode",
                    action,
                    "--heldout-seed",
                    "200003",
                ],
            )
        for split in ["train", "heldout"]:
            run(
                "validate_" + name + "_" + split,
                [
                    "scripts/validate_forced_data.py",
                    "--data",
                    str(data / split),
                    "--out",
                    str(root / name / (split + "_quality.json")),
                    "--require-stationarity"
                    if name == "forced_control"
                    else "--no-require-stationarity",
                ],
            )
        if name == "joystick":
            run(
                "calibrate_joystick",
                [
                    "scripts/calibrate_joystick.py",
                    "--data",
                    str(data / "train"),
                    "--out",
                    str(root / "joystick/calibration.json"),
                ],
            )
    data = Path("data") / campaign_name / "forced_control"
    train = str(data / "train")
    heldout = str(data / "heldout")
    stats = str(data / "train_stats.npz")
    st = np.load(stats)
    deltas = []
    for p in sorted((data / "train").glob("episode_*.npy")):
        x = np.load(p, mmap_mode="r")
        deltas.append(
            np.mean((np.diff(x, axis=0) / (st["std"] + 1e-6)) ** 2, axis=(0, 1, 2))
        )
    scales = np.maximum(3 * np.sqrt(np.mean(deltas, axis=0)), 1e-3)
    common = [
        "--data",
        train,
        "--stats",
        stats,
        "--batch",
        "8" if a.smoke else "32",
        "--hidden-dims",
        *dims,
        "--grain-workers",
        "0",
        "--tb",
        "0",
        "--checkpoint-every",
        "0" if a.smoke else "1000",
    ]
    for seed in [0, 1, 2]:
        for variant in ["cr", "p2"]:
            extra = (
                [
                    "--sobolev-weight",
                    ".01",
                    "--energy-floor",
                    "1e-6",
                    "--gradient-ref",
                    "1.0",
                    "--physics-dx",
                    str(1 / resolution),
                ]
                if variant == "p2"
                else []
            )
            run(
                f"{variant}_seed{seed}",
                [
                    "-m",
                    "fluidgenie.training.train_continuous_residual",
                    *common,
                    "--out",
                    str(root / f"{variant}_seed{seed}"),
                    "--steps",
                    str(cr_steps),
                    "--seed",
                    str(seed),
                    "--unroll-steps",
                    "4",
                    "--residual-scales",
                    *[str(x) for x in scales],
                    *extra,
                ],
            )
    for seed in [0, 1, 2]:
        run(
            f"fm_seed{seed}",
            [
                "-m",
                "fluidgenie.training.train_flow_matching",
                *common,
                "--out",
                str(root / f"fm_seed{seed}"),
                "--steps",
                str(fm_steps),
                "--seed",
                str(seed),
                "--physics-dx",
                str(1 / resolution),
            ],
        )
    horizons = ["8", "16"] if a.smoke else ["40", "80", "118"]
    for seed in [0, 1, 2]:
        run(
            f"benchmark_seed{seed}",
            [
                "scripts/benchmark_repaired_models.py",
                "--train",
                train,
                "--heldout",
                heldout,
                "--stats",
                stats,
                "--cr-checkpoint",
                str(root / f"cr_seed{seed}/latest"),
                "--fm-checkpoint",
                str(root / f"fm_seed{seed}/latest"),
                "--out",
                str(root / f"benchmark_seed{seed}"),
                "--horizons",
                *horizons,
                "--episodes",
                str(heldout_episodes),
            ],
        )
    # Screen the physics variant over the identical horizons and random seeds.
    for seed in [0, 1, 2]:
        run(
            f"p2_benchmark_seed{seed}",
            [
                "scripts/benchmark_repaired_models.py",
                "--train",
                train,
                "--heldout",
                heldout,
                "--stats",
                stats,
                "--cr-checkpoint",
                str(root / f"p2_seed{seed}/latest"),
                "--fm-checkpoint",
                str(root / f"fm_seed{seed}/latest"),
                "--out",
                str(root / f"p2_benchmark_seed{seed}"),
                "--horizons",
                *horizons,
                "--episodes",
                str(heldout_episodes),
            ],
        )
    data = Path(str(Path("data") / campaign_name / "joystick"))
    train = str(data / "train")
    heldout = str(data / "heldout")
    stats = str(data / "train_stats.npz")
    st_common = [
        "--data",
        train,
        "--stats",
        stats,
        "--batch",
        "4" if a.smoke else "8",
        "--model-dim",
        "64" if a.smoke else "256",
        "--num-blocks",
        "2" if a.smoke else "4",
        "--num-heads",
        "4" if a.smoke else "8",
        "--grain-workers",
        "0",
        "--tb",
        "0",
        "--checkpoint-every",
        "0" if a.smoke else "1000",
    ]
    run(
        "lam",
        [
            "-m",
            "fluidgenie.training.train_lam",
            *st_common,
            "--out",
            str(root / "lam"),
            "--steps",
            "20" if a.smoke else "6000",
            "--num-latents",
            "8",
            "--latent-dim",
            "32",
            "--seq-len",
            "4",
        ],
    )
    run(
        "lam_discovery",
        [
            "scripts/evaluate_lam_discovery.py",
            "--checkpoint",
            str(root / "lam/latest"),
            "--calibration-dir",
            train,
            "--test-dir",
            heldout,
            "--stats",
            stats,
            "--out",
            str(root / "lam_discovery.json"),
            "--frames",
            "32" if a.smoke else "64",
        ],
    )
    discovery = json.loads((root / "lam_discovery.json").read_text())
    if not discovery["discovery_gate_passed"] or not discovery.get(
        "five_action_mapping_gate", False
    ):
        (root / "downstream_gate.json").write_text(
            json.dumps(
                {
                    "state": "scientific_gate_failed",
                    "reason": "Unsupervised LAM discovery failed the predeclared gate; downstream conditioned training and semantic demos were not run.",
                },
                indent=2,
            )
            + "\n"
        )
        print(
            "SCIENTIFIC_GATE_FAILED: LAM discovery; stopping without retries",
            flush=True,
        )
        return
    run(
        "action_fm",
        [
            "-m",
            "fluidgenie.training.train_flow_matching",
            "--data",
            train,
            "--stats",
            stats,
            "--out",
            str(root / "action_fm"),
            "--steps",
            str(fm_steps),
            "--batch",
            "8" if a.smoke else "32",
            "--hidden-dims",
            *dims,
            "--use-lam",
            "--lam-ckpt",
            str(root / "lam/latest"),
            "--lam-discovery",
            str(root / "lam_discovery.json"),
            "--grain-workers",
            "0",
            "--tb",
            "0",
            "--checkpoint-every",
            "0" if a.smoke else "1000",
            "--physics-dx",
            str(1 / resolution),
        ],
    )
    run(
        "tokenizer",
        [
            "-m",
            "fluidgenie.training.train_tokenizer_st",
            *st_common,
            "--out",
            str(root / "tokenizer"),
            "--steps",
            str(st_steps),
            "--codebook",
            "64" if a.smoke else "256",
            "--embed",
            "32" if a.smoke else "64",
            "--seq-len",
            "4",
            "--bg-thresh",
            "0",
        ],
    )
    run(
        "st_dynamics",
        [
            "-m",
            "fluidgenie.training.train_dynamics_st",
            "--data",
            train,
            "--stats",
            stats,
            "--out",
            str(root / "st_dynamics"),
            "--vq-ckpt",
            str(root / "tokenizer/latest"),
            "--model",
            "st_maskgit",
            "--steps",
            str(dyn_steps),
            "--batch",
            "4" if a.smoke else "8",
            "--context",
            "2",
            "--codebook",
            "64" if a.smoke else "256",
            "--embed",
            "32" if a.smoke else "64",
            "--d-model",
            "64" if a.smoke else "256",
            "--n-heads",
            "4" if a.smoke else "8",
            "--n-layers",
            "2" if a.smoke else "4",
            "--grain-workers",
            "0",
            "--tb",
            "0",
            "--checkpoint-every",
            "0" if a.smoke else "1000",
            "--tok-model-dim",
            "64" if a.smoke else "256",
            "--tok-num-blocks",
            "2" if a.smoke else "4",
            "--tok-num-heads",
            "4" if a.smoke else "8",
            "--use-lam",
            "--lam-ckpt",
            str(root / "lam/latest"),
            "--lam-model-dim",
            "64" if a.smoke else "256",
            "--lam-latent-dim",
            "32",
            "--lam-num-latents",
            "8",
            "--lam-num-blocks",
            "2" if a.smoke else "4",
            "--lam-num-heads",
            "4" if a.smoke else "8",
            "--bg-thresh",
            "0",
        ],
    )
    run(
        "st_rollout",
        [
            "scripts/evaluate_repaired_st.py",
            "--tokenizer",
            str(root / "tokenizer/latest"),
            "--dynamics",
            str(root / "st_dynamics/latest"),
            "--lam",
            str(root / "lam/latest"),
            "--train",
            train,
            "--heldout",
            heldout,
            "--stats",
            stats,
            "--out",
            str(root / "st_rollout"),
            "--horizon",
            "4" if a.smoke else "40",
        ],
    )
    run(
        "counterfactual",
        [
            "scripts/benchmark_counterfactual_steering.py",
            "--checkpoint",
            str(root / "action_fm/latest"),
            "--stats",
            stats,
            "--out",
            str(root / "counterfactual"),
            "--resolution",
            str(resolution),
            "--horizon",
            "8" if a.smoke else "40",
        ],
    )
    run(
        "st_counterfactual",
        [
            "scripts/benchmark_st_counterfactual.py",
            "--tokenizer",
            str(root / "tokenizer/latest"),
            "--dynamics",
            str(root / "st_dynamics/latest"),
            "--lam",
            str(root / "lam/latest"),
            "--stats",
            stats,
            "--discovery",
            str(root / "lam_discovery.json"),
            "--simulator-branches",
            str(root / "counterfactual/counterfactual_arrays.npz"),
            "--out",
            str(root / "st_counterfactual"),
        ],
    )
    run(
        "showcase",
        [
            "scripts/export_steering_showcase.py",
            "--checkpoint",
            str(root / "action_fm/latest"),
            "--episode",
            str(data / "heldout/episode_000000.npy"),
            "--stats",
            stats,
            "--out",
            str(root / "square_steering.gif"),
        ],
    )
    print("REPAIR_CAMPAIGN_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
