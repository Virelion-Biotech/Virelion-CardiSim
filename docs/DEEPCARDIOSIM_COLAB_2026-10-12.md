# Reviewed bounded DeepCardioSim benchmark

The returned Colab run completed all three seeds on CUDA against six producer-checksummed processed shards. This measures the native CardiGINO geometry-to-activation surrogate; it does not establish empirical validation of CardiSim's phenotype simulator, clinical validation, or a reproduction of the DeepCardioSim paper.

## Results and verification

| Seed | Test MAE | Test RMSE | Test R² | Validation-selected epoch |
|---|---:|---:|---:|---:|
| 12130875 | 5.7036 | 9.1800 | 0.8553 | 24 |
| 1729 | 5.4833 | 8.7829 | 0.8675 | 27 |
| 42 | 5.1938 | 8.3808 | 0.8794 | 29 |
| Training-mean baseline | 18.5428 | 24.1324 | -0.0003 | — |

Mean seed RMSE is 8.7813 (sample SD 0.3996), a 63.6% reduction relative to the training-mean baseline. This is an average of three pooled-node RMSEs, not a new pooled metric or an uncertainty interval over independent studies. Target units have not been independently verified here.

The 288 selected cases are split 202/43/43. Case membership and recorded geometry fingerprints are disjoint across partitions. All 288 fingerprints are distinct, but these sampled hashes are not rotation invariant and do not establish independent anatomical subjects or universal geometry equivalence. All cases have 4,804 source nodes, with 4,096 selected for this benchmark. The split remains a bounded numerical evaluation, not independent anatomical qualification.

Review checked every non-manifest exported payload hash, per-seed metric completion receipts, agreement between summary and seed files, test membership, partition coverage, fingerprint disjointness and validation-only best-epoch selection. Original source pin: `b42e3ba1c9141f762be1d1bf3cb2df68e97baab2`. Raw inputs and trained weights were excluded from the uploaded ZIP; their receipts/hashes are retained, but neither predictions nor weight files could be independently rerun here. `state/best-*.pt` remains on the user's Drive; no trained model has been deployed or substituted for existing artifacts.

The original ZIP stored `ARTIFACT_HASHES.json` twice and included its stale self-hash. Both manifest copies are identical; all other checksummed payloads pass. Original results are imported without changing their serialization or values. `IMPORT_REVIEW.json` records the defect and ZIP identity; `PAYLOAD_SHA256.json` hashes the imported review payload without self-reference.

Artifacts: [`artifacts/deepcardiosim_colab_2026_10_12`](../artifacts/deepcardiosim_colab_2026_10_12), including source snapshots, exact launchers, protocol, split, seed results, completion receipts and post-hoc case diagnostics. The results are worthwhile to commit because they exceed the baseline on this scoped benchmark; no production defaults or validation claims are changed.

## What the errors support doing next

These are hypotheses for new experiments, not measured improvements.

1. **Retain all 4,804 nodes.** Every case lost 708 nodes (14.7%) during uniform sampling. Full inputs require only 17.3% more points than the current 4,096 cap. In particular, uniform input sampling can drop sparse pacing flags; quantify retained/dropped pacing nodes before concluding this occurred. Keep full input geometry/features and distinguish output-query subsampling from input subsampling. Measure T4 memory/time rather than assuming it fits.
2. **Extend training under validation control.** The best epochs were 24, 27 and 29 of 30; validation improved from epoch 20 to the selected epoch in all seeds. A predefined 60-epoch budget with early stopping and a validation-triggered learning-rate schedule is justified as an experiment. This does not prove longer training will improve generalization. Keep checkpoint/optimizer/RNG resume semantics and do not choose budgets from test performance.
3. **Investigate case failures and subgroup coverage.** On mean case RMSE across the three seeds, 41/43 cases beat the baseline; cases 83 and 114 do not. Cases 216, 252, 12 and 83 have the largest average model errors. Source shard/case IDs are in `case_error_review.csv`. Inspect input ranges, pacing representation, geometry and target distributions using training/validation cases before defining a remedy. These test diagnostics are post-hoc and must not become tuning objectives.
4. **Use stronger comparisons.** Add a training-fitted feature/coordinate regression and a pointwise neural baseline to separate operator benefit from a weak constant baseline. Evaluate controlled changes on the same validation partition with matched seeds and compute paired geometry-level differences. The existing geometry RMSE bootstrap does not by itself give a paired confidence interval for improvement over baseline.
5. **Audit physical scale and qualification.** CardiGINO normalizes each geometry's bounding box for its latent grid. A future ablation can supply physical extent separately to test whether useful length-scale information is being lost. Audit units, coordinate conventions, conductivity/fiber features, and anatomical IDs first. Qualification needs a genuinely unused case/geometry cohort; the reported test set is now a diagnostic/regression set, not a repeatedly tunable holdout.

Stage these changes one at a time. Freeze selection and stopping rules before running the new evaluation. Increasing grid/mode/channel capacity can follow a validated full-input ablation; the present results alone do not justify silently enlarging the model.

## Export repair

`scripts/export_colab_results.py ROOT OUTPUT.zip` implements the same no-raw-data/no-weight export policy, omits the manifest from its own inventory, writes it exactly once, and replaces the output ZIP atomically. Its regression test exports the same directory twice and verifies every payload hash and exclusion. The archived original launchers remain unchanged for reproducibility; this repair is for subsequent exports.
