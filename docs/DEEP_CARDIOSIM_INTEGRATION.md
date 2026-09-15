# DeepCardioSim integration plan

## Purpose

DeepCardioSim is retained as an external scientific reference, not copied wholesale into CardiSim. The repository is MIT licensed, while at least one model file explicitly attributes its implementation to a separate upstream project. CardiSim therefore records provenance and implements its own interfaces.

## Pinned reference

- Repository: https://github.com/ehsanngh/DeepCardioSim
- Pinned reference commit: `a0b271a9aee84fd7b6299602dd4dbf8a3a01698f`
- Dataset: https://zenodo.org/records/17651628
- Dataset API: https://zenodo.org/api/records/17651628
- Paper: Naghavi E, Wang H, Ziaei-Rad V, et al. *Rapid prediction of cardiac activation in the left ventricle with geometric deep learning: a step towards cardiac resynchronization therapy planning*. npj Digital Medicine. 2026;9:225.
- DOI: 10.1038/s41746-026-02399-7

## Acquisition map

| Reference component | CardiSim treatment | Validation target |
| --- | --- | --- |
| GINO cardiac model | Native CardiGINO implementation | Synthetic activation-time prediction |
| Graph-UNet cardiac model | Native CardiGNN implementations | Geometry-aware baseline against GINO |
| EP data processor | Native geometry/feature preprocessing contract | Shape-safe transforms and inverse transforms |
| Mesh dataset loader | Provenance-aware CardiSim loader | Deterministic sample loading |
| FEM-generated dataset | External dataset; store metadata/checksums, not bulk bytes | Published benchmark reproduction |
| 2D/3D examples | Convert useful cases to small fixtures | CI smoke tests |
| FEniCS container definitions | Reference reproducibility pattern | Simulator provenance |
| neural operator core | Native bounded spectral operator for first CardiGINO benchmark | Reproducible model construction |

## Artifact inventory and retrieval

The repository does not guess Zenodo filenames or checksums. `scripts/inventory_deepcardiosim.py` queries the Zenodo record API and records the returned artifact keys, sizes, download links, and checksum algorithm/digest. With `--download`, artifacts are downloaded and verified before they are retained.

The verified 2026-09-14 inventory contains six processed training shards (`data_chunk_001.pt` through `data_chunk_006.pt`), plus the published `LVmeans.zip`, `realLVs.zip`, and the underlying multipart `data_npy` archive. The authoritative checksum values are stored in `data/references/deepcardiosim_artifacts.json`; the bulk dataset remains outside Git.

Examples:

```bash
python scripts/inventory_deepcardiosim.py --manifest artifacts/deepcardiosim_manifest.json
python scripts/inventory_deepcardiosim.py --download --artifact realLVs.zip --output artifacts/deepcardiosim
python scripts/inventory_deepcardiosim.py --download --artifact data_chunk_001.pt --output artifacts/deepcardiosim
```

## Published-data adapter

`cardisim.deepcardiosim_data.DeepCardioSimSample` is the canonical NumPy representation used by CardiSim. It supports the upstream mapping/NPY layout and published VTK real-LV cases. The VTK adapter preserves the published field names and semantics:

- `input_geom`: point coordinates `(N, 3)`;
- `a`: pacing flag, isotropic conductivity, and 3-component fiber direction `(N, 5)`;
- `y`: activation time `(N, 1)`;
- pacing-neighborhood propagation uses the published radius of `0.75` coordinate units.

Missing optional VTK inputs are zero-filled as in the pinned upstream preprocessing, while activation time is required.

## CardiGNN benchmark

`cardisim.cardignn.CardiGNN` is a native geometry-aware message-passing model with a selectable GraphSAGE baseline. It does not copy the upstream Graph-UNet implementation. Both variants preserve the published GNN data contract: five pointwise features plus three spatial coordinates form the 8-channel model input, and the model predicts one activation-time value per node.

The geometry-aware default makes relative edge displacement and edge distance explicit in every message. Crucially, `graph_pos` remains in raw physical coordinates for radius-based graph construction, while standardized coordinates are used only as model features. This prevents normalization from silently changing the meaning of the published `r=0.5` neighborhood radius.

The benchmark runner uses deterministic case-level shuffling and a train/validation/test split. Normalization statistics are fitted only on training cases. The validation set controls model selection and early stopping; the held-out test set is evaluated only after the best validation epoch is selected. A training-set-mean baseline is reported for comparison.

The revised spatial CardiGNN benchmark produced a held-out test R² of approximately 0.495 on 128 cases, with best validation R² approximately 0.564 at epoch 13. These numbers are an engineering benchmark result, not a claim of matching the published paper.

Examples:

```bash
python scripts/train_cardignn.py \
  --shard artifacts/deepcardiosim/data_chunk_001.pt \
  --max-samples 128 \
  --epochs 20 \
  --architecture spatial \
  --hidden 64 \
  --patience 5
```

The original plain GraphSAGE baseline remains selectable with `--architecture sage`.

The GNN dependencies are optional:

```bash
pip install -e '.[gnn]'
```

PyTorch Geometric documents `torch_geometric` as the basic package and notes limitations around capped CPU radius-graph construction; substantive spatial experiments should therefore use a GPU when available. citeturn243889search0turn243889search6

## CardiGINO benchmark

`cardisim.cardigino.CardiGINO` is a native, bounded geometry-informed neural operator. It follows the important structural idea behind GINO: pointwise functions on arbitrary coordinates are lifted into a regular 3-D latent field, global spectral operator blocks process that latent field, and the latent representation is queried back at arbitrary output coordinates. The current implementation intentionally avoids requiring `neuraloperator`, Open3D, or torch-scatter for the first reproducible benchmark.

The implementation uses differentiable trilinear splatting from point features into a regular cube, low-frequency 3-D Fourier convolutions plus local pointwise convolutions in latent space, and trilinear query back to requested points. The upstream GINO exposes analogous input-GNO, latent FNO, and output-GNO stages; the CardiSim implementation is a native approximation of that data flow rather than a source-code reproduction. citeturn566946view0turn566946view1

The native GINO dependency is optional:

```bash
pip install -e '.[gino]'
```

For a clean Colab runtime, clone the repository first, then install the optional dependency and run the bounded trainer:

```bash
git clone --depth 1 https://github.com/Virelion-Biotech/Virelion-CardiSim.git
cd Virelion-CardiSim
python -m pip install -q -e '.[gino]'
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
python scripts/inventory_deepcardiosim.py \
  --download \
  --artifact data_chunk_001.pt \
  --output artifacts/deepcardiosim
python scripts/train_cardigino.py \
  --shard artifacts/deepcardiosim/data_chunk_001.pt \
  --max-samples 64 \
  --epochs 10 \
  --hidden 32 \
  --grid-size 16 \
  --modes 8 8 8 \
  --spectral-layers 4 \
  --patience 4
```

Do not call `baseline_validation`, `final_test`, or `best_epoch` as Python variables. They are fields printed by the trainer's final JSON report and are also written to `artifacts/cardigino/metrics.json`.

Do not download all six training shards for the first controlled experiment. Start with one shard and keep benchmark output separate from Git-tracked source.

## Colab boundary

No GPU is required for repository CI, schema tests, inventory, or the real-LV parsing smoke test. A GPU is strongly preferred for CardiGINO because its regular 3-D latent field and FFT blocks scale with the cube resolution and channel width.

The first CardiGINO run should be a 64-case, 16^3-grid smoke benchmark. It is intended to catch installation, tensor-shape, FFT, splatting, interpolation, and training-loop failures before any larger experiment. Only after that passes should the grid or case budget be increased.

## Scientific benchmark contract

The benchmark should report:

1. target-field shape and finite-value checks;
2. normalization/inverse-normalization round-trip error;
3. reference GNN/GINO configuration metadata;
4. absolute and relative field errors once model weights are available;
5. runtime and peak memory;
6. resolution-shift results on higher-density meshes;
7. robustness to the perturbations used by the published study;
8. separate evaluation on real LV geometries;
9. a machine-readable report suitable for CardiEval.

The published GINO implementation uses GNO stages around a latent FNO representation and supports arbitrary output query points; those are the key structural axes preserved in the CardiSim native operator. citeturn566946view0turn566946view1

## Provenance rules

- Do not copy DeepCardioSim source files into CardiSim until the exact source and license of each file are recorded.
- Keep external dataset bytes out of Git unless a dataset license explicitly permits redistribution and repository size is appropriate.
- Pin external references by immutable commit or dataset version when possible.
- Preserve the upstream copyright/license notices for any substantial MIT-licensed code that is eventually incorporated.
- Do not represent a reproduction benchmark as an original Virelion result.

## Implementation order

### Phase 1 — complete

Added provenance records and a dependency-light reference API. Fixed package metadata so the declared Python package license matches the AGPL-3.0 license file.

### Phase 2 — complete

Implemented a NumPy-only `EPPreprocessor` and `UnitGaussianNormalizer` that establish the geometry/feature/target normalization contract and per-geometry Cartesian query-grid construction without depending on DeepCardioSim internals.

### Phase 3 — complete

Added runtime Zenodo artifact inventory/checksum retrieval and a deterministic CI fixture. The fixture is intentionally independent of network availability; live dataset retrieval remains an explicit acquisition operation.

### Phase 4 — complete

Implemented the published-data adapter, VTK real-LV smoke path, geometry-aware CardiGNN baseline, corrected train/validation/test benchmark split, and native bounded CardiGINO operator with clean-runtime trainer and tensor-level CI coverage.

### Phase 5

Promote the benchmark into CardiEval with synthetic, resolution-shift, perturbation, and real-geometry tracks.

## Current status

The provenance boundary, preprocessing layer, acquisition script, verified Zenodo inventory, VTK real-LV adapter, native CardiGNN benchmark, and native CardiGINO benchmark path are committed. The spatial CardiGNN has a useful first held-out result and is now the comparison baseline for CardiGINO. The next substantive gate is one fresh-runtime Colab GPU smoke run of `train_cardigino.py` before scaling the operator or attempting closer reproduction of the published GINO configuration.
