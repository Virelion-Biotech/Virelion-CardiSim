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
| GINO cardiac model | Native CardiGINO implementation later | Synthetic activation-time prediction |
| Graph-UNet cardiac model | Native CardiGNN implementation later | Baseline against GINO |
| EP data processor | Native geometry/feature preprocessing contract | Shape-safe transforms and inverse transforms |
| Mesh dataset loader | Replace with provenance-aware loader | Deterministic sample loading |
| FEM-generated dataset | External dataset; store metadata/checksums, not bulk bytes | Published benchmark reproduction |
| 2D/3D examples | Convert useful cases to small fixtures | CI smoke tests |
| FEniCS container definitions | Reference reproducibility pattern | Simulator provenance |
| neural operator core | Prefer maintained upstream dependencies where possible | Reproducible model construction |

## Artifact inventory and retrieval

The repository does not guess Zenodo filenames or checksums. `scripts/inventory_deepcardiosim.py` queries the Zenodo record API and records the returned artifact keys, sizes, download links, and checksum algorithm/digest. With `--download`, every returned artifact is downloaded and its checksum is verified before it is retained.

Example:

```bash
python scripts/inventory_deepcardiosim.py --manifest artifacts/deepcardiosim_manifest.json
python scripts/inventory_deepcardiosim.py --download --output artifacts/deepcardiosim
```

The generated manifest belongs in the local/reproducibility artifact area rather than Git when it contains transient download metadata. Bulk dataset bytes are intentionally excluded from the repository.

## CI fixture

`tests/fixtures/deepcardiosim_ep_smoke.json` is a tiny deterministic contract fixture derived from the published task schema. It is explicitly **not** claimed to be a downloaded DeepCardioSim sample. CI checks geometry/feature/target shape, finite values, reversible normalization, query-grid construction, and the pinned provenance record.

## Scientific benchmark contract

The first CardiSim benchmark should consume one or more published samples and report:

1. target-field shape and finite-value checks;
2. normalization/inverse-normalization round-trip error;
3. reference GNN/GINO configuration metadata;
4. absolute and relative field errors once model weights are available;
5. runtime and memory measurements;
6. resolution-shift results on higher-density meshes;
7. a machine-readable report suitable for CardiEval.

The published study compares GNN and GINO on synthetic cases, finer resolutions, Gaussian input perturbations, and two real-world LV cohorts. CardiEval should preserve these axes rather than collapsing validation to a random train/test split.

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

### Phase 4

Implement CardiGNN and CardiGINO behind common model interfaces, then reproduce a locked reference sample.

### Phase 5

Promote the benchmark into CardiEval with synthetic, resolution-shift, perturbation, and real-geometry tracks.

## Current status

The provenance boundary, preprocessing layer, acquisition script, and CI smoke fixture are committed. The next gate is a real Zenodo artifact inventory followed by loading one verified published artifact and building the first end-to-end model-independent sample adapter.
