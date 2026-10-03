# Geometry-disjoint benchmark

## Purpose

The five-seed benchmark established that the native CardiGINO advantage on the first 128-case subset is stable across model seeds. The next validation question is whether that advantage persists when repeated cardiac geometries are prevented from crossing train, validation, and test partitions.

## Method

`scripts/benchmark_geometry_generalization.py` accepts multiple processed DeepCardioSim shards and constructs a deterministic geometry fingerprint from the point coordinates after centering and isotropic scale normalization. The fingerprint is also independent of point-row ordering and is rounded before hashing.

The fingerprint is deliberately **not rotation invariant** and does not claim topological equivalence. Its purpose is to detect reuse of the same mesh in different cases under a shared coordinate convention.

The runner reports two audits:

1. geometry-group overlap under the ordinary case-random split;
2. geometry-group overlap after grouped assignment.

The grouped split is the required evaluation split. A detected geometry group is assigned wholly to one of train, validation, or test.

Normalization is fitted only on the training partition. CardiGNN retains raw coordinates for physical-radius graph construction while normalized coordinates are used as learned features. CardiGINO receives normalized point features and normalized coordinates under the same training-only statistics used by the paired benchmark.

## Recommended Colab run

Download or verify the six processed DeepCardioSim training shards using the repository inventory script. First run the cheap geometry audit, then run training only after confirming the grouped split has zero shared geometry groups.

```bash
python -m pip install -q -e '.[gnn,gino]'

python scripts/benchmark_geometry_generalization.py \
  --shards \
    artifacts/deepcardiosim/data_chunk_001.pt \
    artifacts/deepcardiosim/data_chunk_002.pt \
    artifacts/deepcardiosim/data_chunk_003.pt \
    artifacts/deepcardiosim/data_chunk_004.pt \
    artifacts/deepcardiosim/data_chunk_005.pt \
    artifacts/deepcardiosim/data_chunk_006.pt \
  --cases-per-shard 64 \
  --max-cases 384 \
  --audit-only \
  --overwrite

python scripts/benchmark_geometry_generalization.py \
  --shards \
    artifacts/deepcardiosim/data_chunk_001.pt \
    artifacts/deepcardiosim/data_chunk_002.pt \
    artifacts/deepcardiosim/data_chunk_003.pt \
    artifacts/deepcardiosim/data_chunk_004.pt \
    artifacts/deepcardiosim/data_chunk_005.pt \
    artifacts/deepcardiosim/data_chunk_006.pt \
  --cases-per-shard 64 \
  --max-cases 384 \
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

The default 64 cases per shard gives balanced representation from all six processed shards using deterministic within-shard sampling rather than taking the first 64 cases. The sampled indices are recorded in the output. Increase `--max-cases` only after verifying the geometry audit and runtime/memory behavior.

## Outputs

The runner writes:

- `artifacts/geometry_benchmark/geometry_audit.json`
- `artifacts/geometry_benchmark/geometry_comparison.json`
- `artifacts/geometry_benchmark/GEOMETRY_BENCHMARK_REPORT.md`

The machine-readable report includes the geometry-group audit, deterministic split indices, training-only normalization statistics, per-seed results, convergence, runtime, and GPU memory.

## Interpretation boundary

A geometry-disjoint split is stronger evidence of transfer beyond repeated meshes than a case-random split, but it is still not a complete test of biological generalization. Rotation, mesh resolution, topology, pacing configuration, conductivity range, and real-patient anatomy remain separate axes.
