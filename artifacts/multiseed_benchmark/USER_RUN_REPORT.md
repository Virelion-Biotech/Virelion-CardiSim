# CardiSim Five-Seed CardiGNN vs CardiGINO Benchmark

This file records the completed five-seed longer-training benchmark run performed in Colab from repository commit `e27dd091bc44b8e3a5dfd4fb4e94e549d816001f`.

## Protocol

- Source: `artifacts/deepcardiosim/data_chunk_001.pt`
- Cases: 128
- Fixed split seed: `12130875`
- Train / validation / test: `90 / 19 / 19`
- Model seeds: `12130875`, `20260917`, `31415927`, `27182818`, `8675309`
- Epoch budget: 60
- Early stopping patience: 12
- Hidden width: 32
- CardiGNN: 4 spatial layers, physical radius 0.5, max 128 neighbors
- CardiGINO: 4 spectral layers, 16^3 latent grid, modes 8x8x8, MLP ratio 2.0
- Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4
- Normalization statistics were fitted on training cases only.
- Test performance was evaluated only after selecting the validation-RMSE checkpoint.

## Test results across five seeds

| Model | MAE mean ± SD (ms) | RMSE mean ± SD (ms) | R² mean ± SD |
|---|---:|---:|---:|
| Constant baseline | 17.6383 | 23.0028 | -0.0070 |
| CardiGNN | 11.6342 ± 0.6725 | 16.4577 ± 0.6697 | 0.4838 ± 0.0425 |
| CardiGINO | 6.6027 ± 0.3178 | 9.9796 ± 0.3789 | 0.8102 ± 0.0144 |

## Per-seed CardiGNN test results

| Seed | MAE (ms) | RMSE (ms) | R² | Best epoch |
|---:|---:|---:|---:|---:|
| 12130875 | 12.1847 | 16.7754 | 0.4644 | 20 |
| 20260917 | 12.5282 | 17.4762 | 0.4187 | 11 |
| 31415927 | 11.2375 | 15.8629 | 0.5211 | 33 |
| 27182818 | 11.1004 | 16.2038 | 0.5003 | 60 |
| 8675309 | 11.1201 | 15.9705 | 0.5146 | 60 |

## Per-seed CardiGINO test results

| Seed | MAE (ms) | RMSE (ms) | R² | Best epoch |
|---:|---:|---:|---:|---:|
| 12130875 | 6.5329 | 9.8039 | 0.8171 | 33 |
| 20260917 | 6.8771 | 10.2739 | 0.7991 | 23 |
| 31415927 | 6.9487 | 10.4388 | 0.7926 | 24 |
| 27182818 | 6.1654 | 9.4919 | 0.8285 | 30 |
| 8675309 | 6.4892 | 9.8896 | 0.8139 | 27 |

## Paired model difference

CardiGINO minus CardiGNN on the same five seeds:

- MAE: `-5.0315 ± 0.6104 ms`
- RMSE: `-6.4781 ± 0.7230 ms`
- R²: `+0.3264 ± 0.0429`

The sign convention is GINO minus GNN, so negative error deltas and positive R² deltas correspond to lower error / higher explained variance for CardiGINO in this paired experiment.

## Convergence

CardiGINO selected epochs 23–33 across all seeds, with mean best epoch 27.4 and SD 4.16. CardiGNN selected epochs 11–60, with mean best epoch 36.8 and SD 22.58; two seeds reached the 60-epoch ceiling.

## Interpretation boundary

This is a controlled five-seed engineering benchmark on a fixed 128-case subset. It supports a robust within-subset difference between the two native architectures, but it does not establish geometry-disjoint generalization, resolution-shift robustness, perturbation robustness, real-LV performance, or full-dataset performance.

The next required experiment is therefore the geometry-grouped evaluation implemented by `scripts/benchmark_geometry_generalization.py`.
