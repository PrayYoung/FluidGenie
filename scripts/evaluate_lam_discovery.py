"""Evaluate unsupervised LAM codes with calibration-only joystick labels and ablations."""

from fluidgenie.data.dataset_npz import load_episode_actions

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
import tyro
from configs.config_io import load_config_json
from fluidgenie.data.splits import assert_disjoint
from fluidgenie.data.preprocess import patchify, unpatchify
from fluidgenie.eval.flow_matching_rollout import _load_stats, _norm
from fluidgenie.eval.action_discovery import discovery_metrics
from fluidgenie.models.lam import LatentActionModel
from fluidgenie.training.checkpoint_utils import load_params


@dataclass
class Args:
    checkpoint: str
    calibration_dir: str
    test_dir: str
    stats: str
    out: str
    frames: int = 64
    batch_size: int = 8


def decode_actions(module, videos, actions):
    h, w = videos.shape[2:4]
    patches = patchify(videos, module.patch_size)
    conditioned = module.action_up(actions) + module.patch_up(patches[:, :-1])
    output = module.decoder(
        conditioned,
        training=False,
        spatial_shape=(
            (h + module.patch_size - 1) // module.patch_size,
            (w + module.patch_size - 1) // module.patch_size,
        ),
    )
    return unpatchify(output, module.patch_size, h, w)


def discovery_windows(episode, seq_len):
    """Cover every transition once, always at trained temporal positions."""
    if seq_len < 2 or len(episode) < seq_len:
        raise ValueError("Episode is shorter than the trained LAM window")
    for end in range(seq_len, len(episode) + 1):
        start = end - seq_len
        # The first window covers its prefix; later windows add only their final transition.
        yield start, episode[start:end], slice(None) if start == 0 else slice(-1, None)


def episode_wide_shuffle(actions, key):
    """Shuffle transition donors across the whole episode, including seq_len=2."""
    permutation = jax.random.permutation(key, len(actions))
    return actions[permutation], permutation


def discrete_code_change_fraction(codes, permutation):
    codes = np.asarray(codes)
    return float(np.mean(codes[np.asarray(permutation)] != codes))


def save_transition_evidence(out, evidence):
    """Retain small aligned code/label arrays so diagnosis needs no GPU rerun."""
    arrays = {}
    for split, columns in evidence.items():
        lengths = {len(values) for values in columns.values()}
        if len(lengths) != 1:
            raise ValueError("Transition evidence columns must have equal lengths")
        for name, values in columns.items():
            arrays[f"{split}_{name}"] = np.asarray(values, dtype=np.int32)
    path = Path(out).with_suffix(".transitions.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def batched_call(fn, *arrays, batch_size=8):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    outputs = []
    for start in range(0, len(arrays[0]), batch_size):
        chunks = [array[start : start + batch_size] for array in arrays]
        count = len(chunks[0])
        padded = [
            jnp.concatenate([a, jnp.repeat(a[-1:], batch_size - count, axis=0)])
            for a in chunks
        ]
        output = fn(*padded)
        outputs.append(jax.tree.map(lambda x: np.asarray(x)[:count], output))
    return jax.tree.map(lambda *xs: np.concatenate(xs), *outputs)


def evaluate(args):
    split = assert_disjoint(args.calibration_dir, args.test_dir)
    saved = load_config_json(args.checkpoint)
    if not saved or not isinstance(saved.get("config"), dict):
        raise ValueError("LAM checkpoint is missing config.json")
    cfg = saved["config"]
    if not cfg or int(cfg.get("seq_len", 0)) < 2:
        raise ValueError("LAM checkpoint must record a trained seq_len >= 2")
    if (cfg.get("architecture_version") or 0) < 2:
        raise ValueError("LAM discovery requires architecture_version >= 2")
    seq_len = int(cfg["seq_len"])
    first = next(
        p
        for p in sorted(Path(args.calibration_dir).glob("episode_*.npy"))
        if not p.name.endswith(".actions.npy")
    )
    stats = _load_stats(args.stats)
    init_video = jnp.asarray(
        _norm(np.asarray(np.load(first)[:seq_len], dtype=np.float32), stats)[None],
        dtype=jnp.float32,
    )
    model = LatentActionModel(
        in_dim=init_video.shape[-1],
        architecture_version=cfg.get("architecture_version", 2),
        **{
            k: cfg[k]
            for k in [
                "model_dim",
                "latent_dim",
                "num_latents",
                "patch_size",
                "num_blocks",
                "num_heads",
                "dropout",
                "codebook_dropout",
            ]
        },
    )
    params = load_params(
        args.checkpoint,
        model.init(jax.random.PRNGKey(0), {"videos": init_video}, training=False)[
            "params"
        ],
    )

    @jax.jit
    def encode_fn(videos):
        result = model.apply(
            {"params": params}, videos, training=False, method=model.vq_encode
        )
        return result["z_q"], result["indices"].reshape(len(videos), seq_len - 1)

    decode_fn = jax.jit(
        lambda videos, actions: model.apply(
            {"params": params}, videos, actions, method=decode_actions
        )
    )
    shuffle_diagnostics = []
    result = {}
    evidence = {}
    errors = {name: [] for name in ["conditioned", "zero", "shuffled"]}
    copy_errors = []
    for name, directory in [
        ("calibration", args.calibration_dir),
        ("test", args.test_dir),
    ]:
        labels = []
        codes = []
        episode_ids = []
        transition_indices = []
        for i, path in enumerate(sorted(Path(directory).glob("episode_*.npy"))):
            if path.name.endswith(".actions.npy"):
                continue
            episode = np.asarray(np.load(path)[: args.frames], dtype=np.float32)
            truth = load_episode_actions(path)
            if len(truth) < len(episode) - 1:
                raise ValueError(
                    "Action labels are shorter than the episode transitions"
                )
            windows = list(discovery_windows(episode, seq_len))
            videos = jnp.asarray(
                np.stack([_norm(window, stats) for _, window, _ in windows]),
                dtype=jnp.float32,
            )
            window_actions, window_codes = batched_call(
                encode_fn, videos, batch_size=args.batch_size
            )
            # Each transition appears once in the canonical sequence, at a trained position.
            ep_actions = jnp.asarray(
                np.concatenate([window_actions[0], window_actions[1:, -1]], axis=0)
            )
            ep_codes = np.concatenate([window_codes[0], window_codes[1:, -1]], axis=0)
            labels.extend(truth[: len(ep_codes)].tolist())
            codes.extend(ep_codes.tolist())
            episode_ids.extend([i] * len(ep_codes))
            transition_indices.extend(range(len(ep_codes)))
            if name == "test":
                shuffled, permutation = episode_wide_shuffle(
                    ep_actions, jax.random.PRNGKey(1000 + i)
                )
                shuffled_windows = jnp.stack(
                    [
                        shuffled[offset : offset + seq_len - 1]
                        for offset, _, _ in windows
                    ]
                )
                shuffle_diagnostics.append(
                    {
                        "episode": path.name,
                        "transitions": len(ep_codes),
                        "donor_changed_fraction": float(
                            np.mean(
                                np.asarray(permutation) != np.arange(len(ep_actions))
                            )
                        ),
                        "code_changed_fraction": discrete_code_change_fraction(
                            ep_codes, permutation
                        ),
                    }
                )
                conditions = [
                    ("conditioned", jnp.asarray(window_actions)),
                    ("zero", jnp.zeros_like(window_actions)),
                    ("shuffled", shuffled_windows),
                ]
                for condition, values in conditions:
                    recon = batched_call(
                        decode_fn, videos, values, batch_size=args.batch_size
                    )
                    per_transition = np.mean(
                        (recon - np.asarray(videos[:, 1:])) ** 2, axis=(2, 3, 4)
                    )
                    errors[condition].extend(
                        np.concatenate(
                            [per_transition[0], per_transition[1:, -1]]
                        ).tolist()
                    )
                copy = np.mean(np.diff(np.asarray(videos), axis=1) ** 2, axis=(2, 3, 4))
                copy_errors.extend(np.concatenate([copy[0], copy[1:, -1]]).tolist())
        result[name] = (np.asarray(labels), np.asarray(codes))
        evidence[name] = {
            "actions": labels,
            "codes": codes,
            "episode_ids": episode_ids,
            "transition_indices": transition_indices,
        }
    metrics = discovery_metrics(
        *result["calibration"],
        *result["test"],
        cfg["num_latents"],
        *[float(np.mean(errors[name])) for name in ["conditioned", "zero", "shuffled"]],
    )
    counts = np.bincount(result["test"][1], minlength=cfg["num_latents"])
    probabilities = counts / max(counts.sum(), 1)
    metrics.update(
        codes_used=int((counts > 0).sum()),
        code_perplexity=float(
            np.exp(
                -np.sum(
                    probabilities[probabilities > 0]
                    * np.log(probabilities[probabilities > 0])
                )
            )
        ),
        causal_ratio_shuffled_over_real=float(
            np.mean(errors["shuffled"]) / max(np.mean(errors["conditioned"]), 1e-12)
        ),
        copy_baseline_mse=float(np.mean(copy_errors)),
    )
    confusion = np.zeros((5, cfg["num_latents"]), dtype=int)
    np.add.at(confusion, result["test"], 1)
    metrics["test_confusion_matrix"] = confusion.tolist()
    metrics.update(
        shuffle_protocol="episode_wide_transition_permutation_v4",
        shuffle_diagnostics=shuffle_diagnostics,
        split=split,
        protocol=vars(args),
        interpretation="A discovery gate is necessary before calling latent codes joystick actions; reconstruction alone is insufficient.",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    evidence_path = save_transition_evidence(out, evidence)
    metrics["transition_evidence"] = evidence_path.name
    out.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 3))
    plot = axis.imshow(confusion, aspect="auto", cmap="Blues")
    axis.set(
        xlabel="Latent code",
        ylabel="Evaluation-only action label",
        title="Held-out LAM code alignment",
    )
    fig.colorbar(plot, ax=axis)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    evaluate(tyro.cli(Args))
