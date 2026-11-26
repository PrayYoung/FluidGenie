# Saved-checkpoint steering evaluation

The shared-state evaluation fix completes the previously blocked benchmark without retraining. All five selected simulator seeds and all 25 action branches completed locally on CPU, with bit-identical initial contexts (maximum difference **0.0**). No new cloud GPU was allocated.

The simulator now initializes, projects, burns in and advances the no-op prefix once, then forks the exact immutable velocity/density/pressure state. The first two frames are checked with `assert_array_equal`; no tolerance was relaxed. Model noise is reset to 12345 for every branch. The unchanged v6 checkpoint, train-only statistics, H40 and Heun-16 are used throughout.

## Results

| Metric | Mean ± episode SD |
|---|---:|
| Localized velocity-response cosine | 0.839892 ± 0.084339 |
| Global velocity-response cosine | 0.793671 ± 0.107736 |
| Vorticity field correlation | 0.836631 ± 0.074556 |
| Density field correlation | 0.741923 ± 0.037748 |
| Physical-state MSE | 0.120680 ± 0.070156 |

| Simulator seed | Local response cosine | Vorticity correlation | Density correlation | State MSE |
|---:|---:|---:|---:|---:|
| 200003 | 0.901905 | 0.890168 | 0.731296 | 0.071206 |
| 200004 | 0.861073 | 0.883541 | 0.745135 | 0.099446 |
| 200005 | 0.702057 | 0.710063 | 0.688631 | 0.243592 |
| 200006 | 0.910029 | 0.869778 | 0.751136 | 0.081952 |
| 200007 | 0.824397 | 0.829603 | 0.793416 | 0.107205 |

Each episode mean gives its branches equal weight; ± is sample SD across five episode means, not a confidence interval or training-seed variation. Response cosine compares action-minus-no-op velocity fields across time and space; the localized version weights the central jet region. No-op is excluded because its response norm is zero. Vorticity and density correlations include all five branches. Density correlation is computed from absolute density fields across all frames and cells, not an isolated action-effect measure. State MSE averages velocity-x, velocity-y and density equally in physical units. Higher cosine/correlation and lower MSE are better.

The measured response directions agree positively with the simulator on average over these fixed cases. This is evidence of control response on the evaluated distribution, not exact long-horizon simulation. The index-selected first-seed final density comparison visibly drifts, especially in no-op and horizontal branches; whole-rollout correlations do not guarantee final-frame shape fidelity.

## Reviewer pass and limits

- The experiment directly tests the context-confound repair: all branches come from one actual simulator state per seed, including pressure, rather than separately regenerated approximations.
- Five seeds (200003–200007), all five actions and the visual seed were fixed in the [evaluation plan](results/lam_v6/evaluation_protocol.json) before examining scores. No seed or negative outcome was discarded.
- Shared model noise reduces stochastic differences between action/no-op comparisons. One fixed model-noise path and one trained LAM/FM pair were evaluated; this does not measure training or sampling uncertainty.
- The held-out data distribution has informed development. This is not a fresh untouched final-test confirmation, a compute-matched architecture comparison, or a per-component ablation.
- Training action holds are 2–4 frames; these 40-frame held actions and 16-frame square commands test longer holds than the training schedule.
- The square GIF is a real 64-step model rollout under four commands. It does not test simulator agreement for switched actions or prove a geometrically square trajectory.
- The original failed GPU campaign remains in `outcome.json.campaign_outcome`; this successful evaluation-only follow-up is recorded separately. The simulator training-data generation path is preserved.
- No new steering-quality acceptance threshold was defined after results. Completion resolves the engineering blocker; physical drift remains a scientific limitation.

## Evidence

- [All per-branch and aggregate metrics](results/lam_v6/counterfactual_summary.json)
- [Checkpoint, source and array hashes](results/lam_v6/steering_provenance.json)
- [CPU tests and media validation](results/lam_v6/steering_validation.json): 93 full-suite passes, plus both focused checks passing after the stronger branch-order assertion; all 164 GIF frames decode.
- [Visual source/rendering manifest](../assets/visuals_manifest.json)
- [Counterfactual final keyframes](../assets/counterfactual_keyframes.png)
- [Training/discovery report and original failure](lam_v6_results.md)

Raw arrays and checkpoints remain local under `outputs/lam_v6`; published evidence contains compact metrics, hashes and actual-data media. Historical numerical artifacts are preserved.

Superseded engineering reports and scratch plans were removed during repository cleanup. The original frozen evaluation design is retained verbatim in the final protocol record. The validation certificate records the earlier 1,432-file inventory check made before cleanup; it is a historical validation count, not the current file count. Final benchmark and steering evidence is retained unchanged.
