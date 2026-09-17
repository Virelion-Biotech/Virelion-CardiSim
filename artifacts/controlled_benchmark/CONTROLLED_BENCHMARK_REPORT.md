# CardiSim Controlled CardiGNN vs CardiGINO Benchmark

Controlled engineering benchmark using the verified DeepCardioSim shard.

## Provenance

- Repository base commit: `6a4aa3f3cbf615d7b130f54a00ebd95ca31a402a`
- Source shard: `artifacts/deepcardiosim/data_chunk_001.pt`
- Random seed: `12130875`
- Maximum cases: `128`
- Device: `cuda`

## Data split

- Train: **90 cases**
- Validation: **19 cases**
- Test: **19 cases**

## Benchmark protocol

- Same source cases and deterministic case split.
- Normalization statistics fitted using training cases only.
- Activation-time target standardized using training cases only.
- Raw physical coordinates retained for CardiGNN graph construction.
- Same optimizer, learning rate, weight decay, seed, epoch budget, and validation-based early stopping.
- CardiGNN uses the native geometry-aware graph architecture.
- CardiGINO uses the native geometry-informed neural operator.
- This is not an exact reproduction of the published DeepCardioSim training protocol.

## Architecture configuration

### CardiGNN

- `in_channels`: `8`
- `hidden_channels`: `32`
- `layers`: `4`
- `radius`: `0.5`
- `max_num_neighbors`: `128`
- `dropout`: `0.0`
- `out_channels`: `1`
- `architecture`: `spatial`

### CardiGINO

- `in_channels`: `5`
- `out_channels`: `1`
- `hidden_channels`: `32`
- `spectral_layers`: `4`
- `grid_size`: `16`
- `modes`: `[8, 8, 8]`
- `mlp_ratio`: `2.0`

## Results

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Constant baseline | 17.638256 | 23.002850 | -0.007050 |
| CardiGNN | 12.183207 | 16.775936 | 0.464374 |
| CardiGINO | 6.669263 | 10.040175 | 0.808146 |

## Validation results

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Constant baseline | 15.995252 | 19.546671 | -0.054689 |
| CardiGNN | 10.003710 | 13.011278 | 0.532675 |
| CardiGINO | 4.501853 | 6.375304 | 0.887803 |

## Selected epochs

- CardiGNN best epoch: **20**
- CardiGINO best epoch: **20**

## Reproducibility files

- `comparison.json` — machine-readable complete benchmark results.
- `cardignn_controlled.pt` — trained CardiGNN checkpoint with configuration, normalization statistics, split indices, and source-shard reference.
- `cardigino_controlled.pt` — trained CardiGINO checkpoint with configuration, normalization statistics, split indices, and source-shard reference.
- `ENVIRONMENT.json` — Python/PyTorch/CUDA/GPU/package provenance.
- `SHA256SUMS.txt` — checksums for all saved benchmark artifacts.
- `RUN_MANIFEST.json` — final provenance and file inventory.

## Scientific interpretation

The benchmark should be interpreted as a controlled engineering comparison on the specified 128-case subset. Test performance is reported descriptively; conclusions about generalization require additional seeds, larger held-out data, and evaluation under distribution-shift and geometry-perturbation conditions.

