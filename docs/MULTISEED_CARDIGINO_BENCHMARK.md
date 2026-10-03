# Multi-Seed CardiGNN vs CardiGINO Benchmark

## Purpose

This benchmark tests whether the first controlled CardiGINO result remains stable when the training/model seed changes and the training budget is extended beyond the initial 20-epoch experiment.

The benchmark fixes the case split and training-derived normalization across all seeds. This isolates seed-to-seed variation in model initialization and training order instead of mixing it with a changing held-out population.

It reports both:

- pooled-node MAE/RMSE/R², matching the existing controlled benchmark;
- case-macro MAE/RMSE/R², which gives every case equal weight regardless of mesh size.

## Runner

Use:

```bash
python scripts/benchmark_cardigino_multiseed.py \
  --shard artifacts/deepcardiosim/data_chunk_001.pt \
  --max-samples 128 \
  --epochs 60 \
  --patience 12 \
  --split-seed 12130875 \
  --seeds 12130875 20260917 31415927 27182818 8675309 \
  --hidden 32 \
  --gnn-layers 4 \
  --gnn-radius 0.5 \
  --gnn-max-neighbors 128 \
  --gino-grid-size 16 \
  --gino-modes 8 8 8 \
  --gino-spectral-layers 4 \
  --gino-mlp-ratio 2.0 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --overwrite
```

A Tesla T4-class GPU is recommended for the 3-D spectral operator. The runner executes seeds sequentially so one GPU is sufficient.

## Why the split seed is fixed

`--split-seed` determines the train/validation/test population and is deliberately kept constant across model seeds. `--seeds` controls model initialization and deterministic training-order shuffling within that fixed population.

The resulting comparison is therefore paired by seed on the same held-out cases.

## Model-selection rule

For every seed:

1. fit feature and target normalization from training cases only;
2. train on the training cases;
3. select the checkpoint with lowest validation RMSE;
4. evaluate the held-out test cases once at that selected checkpoint.

The test set is never used for epoch selection.

## Longer-training gate

The initial 128-case controlled benchmark allowed only 20 epochs. Both CardiGNN and CardiGINO reached epoch 20 without triggering early stopping, so the 20-epoch budget could not establish whether training had converged.

This benchmark defaults to 60 epochs with patience 12. The report explicitly records whether each run hit the epoch limit. If most or all seeds hit the limit, the training budget should be increased before treating the reported performance as a convergence estimate.

## Output

The runner writes:

- `multiseed_comparison.json` — complete machine-readable results, histories, split, normalization, configuration, and per-seed measurements;
- `MULTISEED_BENCHMARK_REPORT.md` — human-readable summary;
- optional `checkpoints/` — per-seed checkpoints only when `--save-checkpoints` is supplied.

Checkpoints are disabled by default because five CardiGINO checkpoints can consume hundreds of megabytes. Metrics and histories should be committed to Git by default; large checkpoint archives can be handled separately when needed.

## Interpretation

A stable CardiGINO advantage requires the paired test advantage to persist across seeds with reasonable dispersion, not merely a single low-error run. Case-macro metrics should be checked alongside pooled metrics because pooled metrics weight cases according to node count.

This benchmark still does not establish geometry-disjoint generalization, resolution-shift robustness, perturbation robustness, or real-LV performance. Those are separate evaluation tracks.
