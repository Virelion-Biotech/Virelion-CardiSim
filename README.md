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
```

## Design principles

1. **Reproducibility:** every stochastic component is seeded and recorded.
2. **Auditability:** model parameters and events are explicit Python objects and serializable configurations.
3. **Bounded biology:** states have interpretable ranges and non-finite values are rejected.
4. **Composability:** challenge events can be combined and interventions can be applied without changing the numerical engine.
5. **Benchmarkability:** population simulation can generate matched control/perturbation cohorts for downstream evaluation.

## Phenotype state vector

The first release models twelve normalized dimensions in `[0, 1]`:

`maturity, contractility, calcium_handling, electrophysiology, metabolism, hypertrophy, fibrosis, inflammation, angiogenesis, viability, oxidative_stress, mitochondrial_health`

These are abstract latent phenotypes, not direct measurements of specific biomarkers.

## Scientific scope and limitations

CardiSim is a **simulation framework**, not a validated physiological digital twin. Parameters are intentionally exposed so that datasets and fitted models can replace defaults later. No preset should be interpreted as quantitatively predictive of a real animal or patient without calibration and external validation.

See [`docs/MODEL.md`](docs/MODEL.md) for the equations, assumptions, and extension points.

## License

MIT. See `LICENSE`.
