# Virelion-CardiSim

CardiSim is a Python simulator for generating synthetic cardiac-cell and cardiac-phenotype trajectories under controlled perturbations. It is intended for software testing, hypothesis generation, benchmark construction, and model evaluation.

## Scope

The simulator provides:

- bounded continuous phenotype states;
- fourth-order Runge–Kutta integration;
- seeded population heterogeneity and cell IDs;
- time-localized challenge events and recovery;
- baseline maturation, myocardial-injury-like, hypoxia, radiation-injury, and electrophysiology/toxicity presets;
- intervention/rescue parameters;
- empirical calibration utilities;
- CardiAtlas metadata/calibration integration;
- trajectory summaries and CSV/JSON output;
- Python API and CLI.

## Model state

The default state contains twelve normalized dimensions:

`maturity, contractility, calcium_handling, electrophysiology, metabolism, hypertrophy, fibrosis, inflammation, angiogenesis, viability, oxidative_stress, mitochondrial_health`

These are latent simulation variables, not direct measurements of biomarkers.

## Installation

```bash
pip install -e .
pip install -e '.[dev]'
```

## Python

```python
from cardisim import CardiacSimulator, SimulationConfig, population_preset

config = SimulationConfig(duration=28, dt=0.25, n_cells=256, seed=42)
sim = CardiacSimulator(config)
result = sim.run(population_preset("mi"))
result.to_csv("mi_population.csv")
```

## CLI

```bash
cardisim simulate --preset mi --cells 256 --days 28 --dt 0.25 --seed 42 --output mi.csv
cardisim derive-targets --expression expression.csv --metadata samples.json --dataset-id GSE185289 --study-id pig_regeneration --output targets.csv
```

## Calibration

The repository contains a public-data calibration panel as metadata. Raw/large expression matrices are not redistributed. Calibration is not considered complete until processed data produce subject-level targets and the fitted dynamics pass held-out validation.

## Integration

- **CardiAtlas:** metadata and source context for calibration.
- **CardiLearn:** learned state representations where appropriate.
- **CardiBench/CardiEval:** synthetic benchmark generation and evaluation.
- **CardiTrace:** simulation provenance.
- **HeartTwin:** scenario execution and trajectory integration.

## Scientific limitations

CardiSim is not a validated physiological model or digital twin. Default parameters are qualitative until externally calibrated. Synthetic trajectories cannot establish that the corresponding biological process will occur in vivo or in vitro.

## Testing

```bash
pytest
```

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.

## Citation

Cite the repository release and all empirical datasets used for calibration.
