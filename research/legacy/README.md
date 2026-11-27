# Retired FluidGenie experiments

This compact record preserves what was tried before the current repaired benchmarks and LAM steering results. It replaces the need to keep every old GPU upload bundle, raw rollout, or intermediate weight file. The reviewed legacy cache bulk has now been retired after preservation checks. The [pruning receipt](pruning_receipt.json) records the deletion scope and post-cleanup validation.

## Start here

- [Experiment registry](experiment_registry.json): configurations, attempted runs, observed parameter values, result links and qualified conclusions.
- [Training parameter table](parameter_scan.csv): the recorded settings for each configuration.
- [Evaluation settings](evaluation_settings.csv): horizons, seeds/episodes, solvers and sampling sweeps.
- [Compact metric summaries](metrics_summary.json.gz): aggregate results and training-log endpoints, preserving separate result versions. Full sanitized reports/configurations/logs are retained locally in `private/evidence.json.gz`.
- [Verification certificate](verification.json): measured coverage, integrity checks and pruning scope.

## What the old investigations established

| Investigation | Settings or question | Recorded outcome | Retention decision |
|---|---|---|---|
| FM0 and sampling controls | One-step samples 1/2/4/8/16/32; recurrent averages 1/4/8/16; fixed-noise control | Old FM0 was overactive. Averaging reduced sampling error but did not fix recurrent calibration. This does not invalidate later repaired FM. | Settings, scalar results and diagnosis retained; old arrays/weights retired. |
| Residual baselines and bounded unrolling | Context 1/2; unroll 1/4; seeds 0/1/2; bounded residual variants | Bounded models improved historical short rollouts but still drifted at long horizons. Evaluation episodes overlapped training data. | Historical baseline; bulk artifacts retired, current repaired experiments protected. |
| Context noise and longer unrolling | Frozen noise sigmas; unroll 8 versus control 4 | Neither tested change passed the original long-horizon gate. A combined intervention was not justified. | Legacy bulk retired; negative finding and tested settings retained. |
| Physics-loss screen | Energy weights 0.002261/0.009044; Sobolev weights 0.00001905/0.00007619; divergence weights 0.00010335/0.0004134 | Energy penalties and nominal divergence penalty were harmful in this setup. Nominal Sobolev penalty initially helped but did not establish a long-horizon solution. | Keep all candidate measurements and the frozen plan, not just favorable results. |
| Historical P2 replication | Additional training seeds with the nominal Sobolev setting | Seed-1 regression and incomplete retrieval left the historical replication unresolved. Cause was not established. | Record as inconclusive/superseded; current repaired P2 results remain separate and protected. |

These are conclusions about the recorded settings and protocols, not proofs that entire research directions are impossible. Runner `SUCCESS` means the command completed, not that a scientific gate passed. Infrastructure/workload failures are recorded separately. The older training-overlap evaluations do not establish unseen-data generalization and must not be compared as matched controls to current results.

## Private source and provenance archive

`private/` is deliberately Git-ignored. Keep it: the original cache is no longer the backing store for these records. It contains the full compact numeric evidence bundle, exact unique source/config versions, dependency locks, run manifests, and small raw dataset-metadata records. This material can contain historical local paths or setup details and must not be published wholesale. All retained objects are content-addressed by SHA-256; the archive is independent of the global Vast cache.

Failed execution records also preserve short error excerpts privately; an infrastructure error is not relabeled as a failed scientific hypothesis.

Per-run manifests are in `private/run_manifests/`; source files point to `private/objects/<sha256>`. `private/index.json` also records recovered sources from earlier verified cleanup. Dataset metadata stored inside object-type NPY records was preserved as raw bytes, never unpickled or executed during this audit. The package intentionally does not retain the old fluid arrays or model tensors needed for exact legacy replay.

## Coverage and limitations

The coverage counts and payload sizes are in [verification.json](verification.json). The original-case archive covers 43 remaining upload bundles and recovery records for previously removed bundles. Six probe/failed/delivery attempts have no complete upload source available; their recorded commands/commits/status are retained, without inventing missing source or outcomes. All source/config files selected from available bundles and earlier preservation records were recovered without gaps. Saved-result configurations are distinguished from captured upload inputs: an input checkpoint configuration is not proof that a run trained it. Some repeated evaluations have multiple result versions; these are not independent replications. Original aggregate reports and historical review notes remain in the private compressed evidence bundle. Missing historical results are not invented.

The dataset fingerprints, source versions, parameters and measurements support understanding and auditing the old work. After raw data and weights are deliberately retired, rerunning those exact old models without regeneration/retraining is not guaranteed. Current final-result data, weights and resume states remain untouched.

The later [LAM v5 diagnosis](../lam_v5_diagnosis.md), [LAM v6 results](../lam_v6_results.md) and [current steering report](../steering_evaluation_results.md) remain separate. They are not replaced by these old protocol results.

## Completed retirement boundary

The original reviewed proposal remains in `private/pruning_manifest.json.gz`; the completed action is recorded in [pruning_receipt.json](pruning_receipt.json) and the private compressed execution ledger. The cleanup removed 3,122 files totaling 19.752 GiB, covering only FluidGenie upload snapshots and downloaded outputs. The user-approved retirement also included the old preparatory snapshot with no recorded instance. Other projects and cloud instances were untouched.

Keep `data/repair_v4`, `data/lam_v5`, `outputs/repair_v4`, `outputs/lam_v6`, current `research/results`, and `assets`. The `lam_v5` name on the current dataset does not make it obsolete: the current LAM uses those inputs. After cleanup, 1,928 protected file hashes were unchanged. All 12 retained models restored on CPU; all 11 existing optimizer/RNG resume states restored and matched their final weights. The parameter-only historical LAM never had a full optimizer state; this cleanup did not remove one.

For future experiments: preserve a verified summary/config/source record and the selected final model/resume state. Retire intermediate checkpoints and upload bundles after successful verification. Historical campaigns that will not be resumed need compact records, not indefinite copies of every tensor and frame.
