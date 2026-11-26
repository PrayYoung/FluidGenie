# LAM v5 diagnosis

This is the historical negative-result diagnosis. The later [training/discovery report](lam_v6_results.md) and [steering evaluation](steering_evaluation_results.md) supersede its proposed next experiment.

The completed job is a valid negative result. No data-corruption, interrupted-training, checkpoint-export, normalization, or transition-label alignment fault was found in this audit. The joint v5 intervention did not solve semantic action discovery. Do not rerun it unchanged.

## Verified evidence

- The 138 delivered files match their saved hashes. The original 30 training episodes and all ten held-out episodes are unchanged. Added training seeds are exactly 30–59; training/held-out seeds and content are disjoint. Training-only statistics recompute correctly.
- CUDA preflight and all 80 GPU regression tests passed. The saved model/optimizer/RNG/usage state is finite at training step 8000; exported latest parameters exactly match the training-state parameters.
- CPU inference exactly reproduces all 630 held-out code assignments summarized by the GPU confusion matrix. Label shifts of -2/-1/0/+1/+2 give 23.9%/35.2%/46.0%/34.8%/24.9% calibrated accuracy. The correct index is best; this is not explained by a simple label offset.
- All eight codes are active; shuffled reconstruction error is 2.177 times the real-code error. The model uses its codes, but the codes are not clean joystick labels.
- Even an inadmissible test-label-based majority remapping reaches only 52.9%. This is a diagnostic ceiling for relabeling the current codes, not a valid reported benchmark. Merely fixing the calibration mapping cannot recover 80%.
- The frozen mapping is also weak on sampled training episodes: 36.5%–68.3% over twelve episodes spanning old and new train seeds. The problem is not solely held-out generalization. Ten same-transition context probes change code in 2/10 cases; this is a small diagnostic sample, not proof of a cause.
- A simple supervised nearest-centroid classifier of two local velocity-change features reaches 86.35% on the same 630 transitions, with centroids fitted only on training data. A state-only linear probe reaches 32.70%, versus 73.81% for pooled velocity changes. These diagnostic methods differ and are not LAM benchmarks; they establish that useful action information is present in state changes.

## Interpretation and fix decision

The leading explanation is a representation/objective mismatch. The encoder receives absolute state patches and learns a quantized representation through next-state reconstruction plus VQ/commitment losses. There is no explicit constraint that the codes ignore state appearance and represent the externally applied joystick action. Better reconstruction therefore does not guarantee better action alignment. This is an inference from the implementation and diagnostics, not an isolated causal finding.

The previous screenshot's suggested causes were hypotheses. V5 jointly changed spatial features, prefix embeddings, training diversity, batch size, temporal context and training steps, so this run cannot identify which individual change caused the regression. Normalized reconstruction MSE also uses different training statistics across runs and is not a directly matched physical-error comparison.

One software observability gap is fixed: future discovery evaluations save compact per-transition labels, codes, episode IDs and transition indices, plus the calibration confusion matrix and calibration mapping purity. The frozen v4 thresholds, mapping procedure, and pass/fail computation are unchanged. This prevents another GPU run merely to recover diagnostic evidence.

Recommended next experiment: a single controlled LAM-only transition-encoder ablation, rather than another full campaign. Keep the v5 data, periodic encoding, batch, sequence length, optimizer budget and frozen gate. Change only encoder inputs to explicit temporal differences while preserving the decoder's original state context. Test temporal causality and invariance to time-constant appearance offsets locally before allocating a GPU. This is a new scientific hypothesis, not a confirmed bug repair. Do not simultaneously add a residual decoder, tune thresholds, retrain CR/P2/FM, or launch downstream FM unless the original gate passes. No new GPU job was launched during this diagnosis.

## Reproduction

```bash
PYTHONPATH=. uv run python scripts/diagnose_lam_checkpoint.py
uv run python scripts/diagnose_joystick_signal.py
PYTHONPATH=. uv run pytest -q tests/test_lam_diagnostics.py
```

Retained evidence: [outcome](results/lam_v5/outcome.json), [alignment comparison](results/lam_v5/diagnostics/alignment_comparison.png), and [physics probe](results/lam_v5/diagnostics/physics_probe.json). Detailed intermediate diagnostic dumps were pruned from the public checkout; the retained outcome, plot and physics probe summarize the verified findings. The commands above regenerate diagnostics from the original local inputs. The diagnostic inference uses saved checkpoints and local data; it requires no remote compute.
