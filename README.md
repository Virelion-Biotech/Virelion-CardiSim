# Virelion-CardiSim

**Virelion-CardiSim** is a transparent, reproducible simulator for generating synthetic cardiac-cell and cardiac-phenotype trajectories under controlled perturbations.

It is designed as an infrastructure component of Virelion's cardiac challenge stack:

`CardiAgent → CardiVex → CardiSim → CardiEval → CardiLearn`

The simulator is deliberately **synthetic and mechanistic**. Its trajectories are useful for software development, stress-testing, hypothesis generation, benchmark construction, and method evaluation; they are not a substitute for experimental measurements.

## What is included

- Coupled continuous cardiac phenotype state model with bounded states.
- Fourth-order Runge–Kutta integration with deterministic seeds.
- Population-level heterogeneity and reproducible cell IDs.
- Time-localized challenge events with onset, duration, magnitude, and recovery.
- Built-in presets for baseline maturation, myocardial infarction-like injury, hypoxia, radiation injury, and electrophysiology/toxicity stress.
- Intervention support for rescue/attenuation perturbations.
- Empirical calibration through regularized latent-dynamics fitting.
- Explicit CardiAtlas calibration bridge and provenance-aware target ingestion.
- Reproducible GEO source helpers and transparent cardiac proxy marker modules.
- Trajectory summaries and health/maturity scores.
- JSON serialization and CSV export.
- Python API and `cardisim` CLI.
- Unit tests and GitHub Actions CI.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest
```

## Minimal Python example

```python
from cardisim import CardiacSimulator, SimulationConfig, population_preset

config = SimulationConfig(duration=28, dt=0.25, n_cells=256, seed=42)
sim = CardiacSimulator(config)
result = sim.run(population_preset("mi"))

print(result.summary())
result.to_csv("mi_population.csv")
```

## CLI

```bash
cardisim simulate --preset mi --cells 256 --days 28 --dt 0.25 --seed 42 --output mi.csv
cardisim derive-targets --expression expression.csv --metadata samples.json --dataset-id GSE185289 --study-id pig_regeneration --output targets.csv
```

## Empirical calibration panel

CardiSim now has a locked public-data calibration panel:

| Accession | Organism | Modality | Role |
|---|---|---|---|
| **GSE185289** | pig | snRNA-seq | maturation, regenerative vs non-regenerative injury, remodeling |
| **GSE240848** | rat | snRNA-seq | acute MI / ischemia-reperfusion |
| **GSE135310** | mouse | scRNA-seq | post-MI inflammation |

These accessions are stored as metadata in `data/reference/calibration_panel.json`; raw/large expression matrices are intentionally not redistributed in the repository. The calibration workflow fetches the public processed source at runtime, records a checksum, derives 12 transparent phenotype proxy targets, and only then fits the latent dynamics.

**Important:** the panel is currently **source-locked, not numerically fit**. No claim of biological calibration is made until processed expression data have been converted to subject-level targets and the fit passes held-out validation.

See [`docs/CALIBRATION_PANEL.md`](docs/CALIBRATION_PANEL.md) for the data policy, calibration gate, and target definition.

## Design principles

1. **Reproducibility:** every stochastic component is seeded and recorded.
2. **Auditability:** model parameters and events are explicit Python objects and serializable configurations.
3. **Bounded biology:** states have interpretable ranges and non-finite values are rejected.
4. **Composability:** challenge events can be combined and interventions can be applied without changing the numerical engine.
5. **Benchmarkability:** population simulation can generate matched control/perturbation cohorts for downstream evaluation.
6. **Evidence discipline:** public datasets are sources, while derived latent targets and fitted parameters are versioned artifacts with provenance.

## Phenotype state vector

The simulator models twelve normalized dimensions in `[0, 1]`:

`maturity, contractility, calcium_handling, electrophysiology, metabolism, hypertrophy, fibrosis, inflammation, angiogenesis, viability, oxidative_stress, mitochondrial_health`

These are abstract latent phenotypes, not direct measurements of specific biomarkers.

## Scientific scope and limitations

CardiSim is **not a validated physiological digital twin**. Default parameters remain qualitative until external calibration and validation are complete. Proxy marker modules are transparent computational constructs and must not be interpreted as established clinical biomarkers.

See [`docs/MODEL.md`](docs/MODEL.md) and [`docs/CALIBRATION_PANEL.md`](docs/CALIBRATION_PANEL.md).

## License

MIT. See `LICENSE`.
