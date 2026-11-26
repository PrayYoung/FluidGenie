"""Summarize frozen three-seed benchmarks and display recorded model rollouts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    horizons = [40, 80, 118]
    report = {"aggregation": "Three training seeds; each evaluated on the same ten held-out episodes. SD is across seed means. MSE averages all three physical channels.",
              "horizons": {}, "source_commit": "51d0b5aa9b12c95cdef1cf89a8f1ffb247be4961"}
    metrics = ["mse_mean", "motion_ratio_of_means", "vorticity_correlation",
               "tke_ratio", "enstrophy_ratio", "divergence_rms", "frozen_fraction"]
    for horizon in horizons:
        models = {}
        for model, prefix, key in [("CR", "benchmark", "cr"), ("P2", "p2_benchmark", "cr"), ("FM", "benchmark", "fm")]:
            seed_means = {metric: [] for metric in metrics}
            for seed in range(3):
                data = json.loads((args.artifacts / f"{prefix}_seed{seed}/benchmark_summary.json").read_text())
                assert data["split"]["disjoint"]
                rows = [r for r in data["episodes"] if r["horizon"] == horizon]
                assert len(rows) == 10
                for metric in metrics:
                    values = [r[key][metric] for r in rows]
                    assert np.isfinite(values).all(), (model, metric)
                    seed_means[metric].append(float(np.mean(values)))
            models[model] = {metric: {"mean": float(np.mean(values)),
                                      "seed_sd": float(np.std(values, ddof=1)),
                                      "seed_means": values}
                             for metric, values in seed_means.items()}
        baseline = json.loads((args.artifacts / "benchmark_seed0/benchmark_summary.json").read_text())
        persistence = float(np.mean([r["persistence_mse"] for r in baseline["episodes"] if r["horizon"] == horizon]))
        for model in models.values():
            model["mse_reduction_vs_persistence"] = 1 - model["mse_mean"]["mean"] / persistence
        report["horizons"][str(horizon)] = {"persistence_mse": persistence, "models": models}
    lam = json.loads((args.artifacts / "lam_discovery.json").read_text())
    report["action_discovery"] = {k: lam[k] for k in ["nmi", "out_of_sample_purity", "active_test_codes", "discovery_gate_passed", "five_action_mapping_gate", "causal_ratio_shuffled_over_real", "conditioned_mse", "zero_code_mse", "shuffled_code_mse", "gate"]}
    report["downstream_gate"] = json.loads((args.artifacts / "downstream_gate.json").read_text())
    report["limits"] = ["Ten shared held-out episodes and three training seeds; no population-wide superiority claim.",
                        "FM uses 15000 training steps versus 3000 for CR/P2 and 16 solver steps at inference; comparisons are not compute matched.",
                        "Motion/energy ratios remain above one at long horizons and vorticity correlation decays.",
                        "The repaired protocol differs from historical data/model settings; historical MSE is not a matched control.",
                        "LAM fails the frozen semantic discovery gate; conditioned models and semantic demos were not run."]
    (args.out / "results_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    colors = {"CR": "#6b7280", "P2": "#d97706", "FM": "#2563eb"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, metric, title in [(axes[0, 0], "mse_mean", "Held-out prediction error · lower is better"),
                              (axes[0, 1], "motion_ratio_of_means", "Predicted / true motion · ideal = 1"),
                              (axes[1, 0], "vorticity_correlation", "Vorticity correlation · ideal = 1")]:
        for model in colors:
            values = [report["horizons"][str(h)]["models"][model][metric] for h in horizons]
            ax.errorbar(horizons, [v["mean"] for v in values], yerr=[v["seed_sd"] for v in values],
                        label=model, color=colors[model], marker="o", capsize=3)
        if metric == "mse_mean":
            ax.plot(horizons, [report["horizons"][str(h)]["persistence_mse"] for h in horizons], "--", color="#991b1b", label="Persistence")
        elif metric == "motion_ratio_of_means":
            ax.axhline(1, color="#16a34a", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=11); ax.set_xlabel("Forecast frames")
        ax.set_xticks(horizons); ax.grid(alpha=.18); ax.legend(fontsize=8)
    confusion = np.asarray(lam["test_confusion_matrix"])
    semantic = np.zeros((5, 5))
    for code, action in enumerate(lam["code_to_action"]):
        semantic[:, action] += confusion[:, code]
    semantic /= semantic.sum(axis=1, keepdims=True)
    ax = axes[1, 1]
    im = ax.imshow(semantic, cmap="Blues", vmin=0, vmax=1)
    labels = ["No-op", "Up", "Down", "Right", "Left"]
    ax.set_xticks(range(5), labels); ax.set_yticks(range(5), labels)
    ax.set_xlabel("Decoded action"); ax.set_ylabel("True action")
    status = "PASS" if lam["discovery_gate_passed"] and lam.get("five_action_mapping_gate", False) else "FAILED"
    ax.set_title(f"LAM purity: {lam['out_of_sample_purity']:.1%} · "
                 f"required: {lam['gate']['purity_min']:.0%} · {status}", fontsize=11)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{semantic[i,j]:.0%}", ha="center", va="center", color="white" if semantic[i,j] > .55 else "#172033", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("Repaired models: three training seeds × ten held-out episodes", fontsize=14)
    fig.supxlabel("Error bars: SD across training-seed means. CR: residual baseline; P2: physics-regularized residual; FM: flow matching.", fontsize=9)
    fig.savefig(args.out / "benchmark_overview.png", dpi=150)
    plt.close(fig)
    # Preselected episode 0 / seed 0; this is a case illustration, not an average.
    predictions = {}
    gt = None
    for model, prefix, key in [("CR", "benchmark", "cr"), ("P2", "p2_benchmark", "cr"), ("FM", "benchmark", "fm")]:
        path = args.artifacts / f"{prefix}_seed0/{key}/episode_000000/h118/rollout_arrays.npz"
        with np.load(path) as z:
            predictions[model] = z["pred"]
            if gt is None:
                gt = z["gt"]
            else:
                assert np.array_equal(gt, z["gt"]), "Matched rollouts need identical truth"
    def vorticity(x):
        return ((np.roll(x[..., 1], -1, axis=2) - np.roll(x[..., 1], 1, axis=2)) -
                (np.roll(x[..., 0], -1, axis=1) - np.roll(x[..., 0], 1, axis=1))) * 32
    fields = {"Simulator truth": vorticity(gt), **{k: vorticity(v) for k, v in predictions.items()}}
    scale = float(np.quantile(np.abs(fields["Simulator truth"]), .99))
    fig, axes = plt.subplots(4, 3, figsize=(9, 10), layout="constrained")
    for row, (name, field) in enumerate(fields.items()):
        for col, index in enumerate([39, 79, 117]):
            ax = axes[row, col]
            image = ax.imshow(field[index], origin="lower", cmap="RdBu_r", vmin=-scale, vmax=scale)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(f"Forecast frame {index + 1}")
            if col == 0: ax.set_ylabel(name)
    fig.colorbar(image, ax=axes, shrink=.5, label="Vorticity · shared color scale")
    fig.suptitle("Actual saved rollouts · held-out episode 0, training seed 0", fontsize=13)
    fig.supxlabel("Illustrative case selected by episode index, without selecting for score.", fontsize=10)
    fig.savefig(args.out / "recorded_rollout_comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps({"models": 9, "heldout_episodes": 10, "training_seeds": 3,
                      "lam_discovery_gate_passed": lam["discovery_gate_passed"]}))


if __name__ == "__main__":
    main()
