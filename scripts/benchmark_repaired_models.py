"""Matched unseen-seed benchmark; load once and slice one maximum-horizon rollout."""

from dataclasses import dataclass
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from configs.config_io import load_config_json
from fluidgenie.data.splits import assert_disjoint, assert_checkpoint_evaluation
from fluidgenie.eval.continuous_residual_rollout import _load_residual_model
from fluidgenie.eval.flow_matching_diagnostics import _load_model
from fluidgenie.eval.flow_matching_rollout import _load_stats, _norm, _denorm
from fluidgenie.eval.samplers.flow_matching import sample_flow_matching_heun
from fluidgenie.eval.metrics import (
    fluid_physics_metrics,
    jsonable,
    episode_physics_metadata,
    compute_rollout_metrics,
    summarize_physics,
)


@dataclass
class Args:
    train: str
    heldout: str
    stats: str
    cr_checkpoint: str
    fm_checkpoint: str
    out: str
    horizons: tuple[int, ...] = (40, 80, 118)
    episodes: int = 10
    ode_steps: int = 16


def main(args):
    if args.episodes < 1 or not args.horizons or min(args.horizons) < 1:
        raise ValueError("Positive episode count and horizons are required")
    split = assert_disjoint(args.train, args.heldout)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    first = np.load(Path(args.heldout) / "episode_000000.npy", mmap_mode="r")
    _, h, w, c = first.shape
    cr, cr_params, cr_saved = _load_residual_model(args.cr_checkpoint, h, w, c)
    fm, fm_params = _load_model(args.fm_checkpoint, h, w, c)
    if int(cr_saved["config"]["context"]) != 2 or fm.action_dim:
        raise ValueError(
            "This matched benchmark requires context=2 and an unconditional FM"
        )
    for saved in [cr_saved, load_config_json(args.fm_checkpoint)]:
        assert_checkpoint_evaluation(saved["config"], args.heldout)
    stats = _load_stats(args.stats)
    cr_step = jax.jit(
        lambda ctx: ctx[:, -1] + cr.apply({"params": cr_params}, ctx, training=False)
    )
    fm_step = jax.jit(
        lambda ctx, key: sample_flow_matching_heun(
            fm, fm_params, ctx, key, args.ode_steps
        )
    )
    rows = []
    maximum = max(args.horizons)
    for episode in range(args.episodes):
        path = Path(args.heldout) / f"episode_{episode:06d}.npy"
        fields = np.asarray(np.load(path), dtype=np.float32)
        if fields.shape[1:] != (h, w, c) or len(fields) < maximum + 2:
            raise ValueError(
                "Episode shape or maximum horizon does not match benchmark"
            )
        initial = jnp.asarray(_norm(fields[:2], stats)[None], dtype=jnp.float32)
        cr_context = fm_context = initial
        key = jax.random.PRNGKey(1001 + episode)
        predictions = {"cr": [], "fm": []}
        for _ in range(maximum):
            key, step_key = jax.random.split(key)
            cr_next = cr_step(cr_context)
            fm_next = fm_step(fm_context, step_key)
            for name, value in [("cr", cr_next), ("fm", fm_next)]:
                predictions[name].append(_denorm(np.asarray(value[0])[None], stats)[0])
            cr_context = jnp.concatenate([cr_context[:, 1:], cr_next[:, None]], axis=1)
            fm_context = jnp.concatenate([fm_context[:, 1:], fm_next[:, None]], axis=1)
        predictions = {name: np.asarray(values) for name, values in predictions.items()}
        for horizon in args.horizons:
            gt = fields[2 : 2 + horizon]
            row = dict(
                episode=episode,
                horizon=horizon,
                persistence_mse=float(np.mean((gt - fields[1]) ** 2)),
            )
            for name, prediction in predictions.items():
                pred = prediction[:horizon]
                metrics = compute_rollout_metrics(gt, pred, view="density")
                physics = fluid_physics_metrics(
                    gt, pred, last_context=fields[1], **episode_physics_metadata(path)
                )
                row[name] = {
                    key: float(metrics[key])
                    for key in ("mse_mean", "motion_ratio_of_means", "frozen_fraction")
                }
                row[name].update(summarize_physics(physics))
                out = root / name / f"episode_{episode:06d}" / f"h{horizon}"
                out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out / "rollout_arrays.npz", gt=gt, pred=pred)
                (out / "physics_metrics.json").write_text(
                    json.dumps(jsonable(physics), indent=2, allow_nan=False) + "\n"
                )
            rows.append(row)
    report = dict(
        protocol=vars(args),
        split=split,
        episodes=rows,
        sampling_protocol="One maximum-horizon trajectory per model and episode; shorter horizons share its prefix.",
        conclusion="Quality is measured; training completion is not automatic promotion.",
    )
    (root / "benchmark_summary.json").write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    main(tyro.cli(Args))
