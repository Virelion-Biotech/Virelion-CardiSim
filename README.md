# Virelion-CardiSim

CardiSim is a Python simulator for generating synthetic cardiac-cell and cardiac-phenotype trajectories under controlled perturbations. It is intended for software testing, hypothesis generation, benchmark construction, and model evaluation.

## What it contains

- Bounded continuous phenotype states.
- Fourth-order Runge–Kutta integration.
- Seeded population heterogeneity and cell IDs.
- Time-localized challenge events and recovery.
- Presets for baseline maturation, myocardial-injury-like states, hypoxia, radiation injury, and electrophysiology/toxicity.
- Intervention/rescue parameters.
- Empirical calibration utilities.
- Trajectory summaries and CSV/JSON output.
- Python API and CLI.

The default state contains twelve normalized dimensions: `maturity`, `contractility`, `calcium_handling`, `electrophysiology`, `metabolism`, `hypertrophy`, `fibrosis`, `inflammation`, `angiogenesis`, `viability`, `oxidative_stress`, and `mitochondrial_health`. These are latent simulation variables, not direct biomarker measurements.

## Installation

```bash
pip install -e .
pip install -e '.[dev]'
```

## Usage

Python:

```python
from cardisim import CardiacSimulator, SimulationConfig, population_preset

config = SimulationConfig(duration=28, dt=0.25, n_cells=256, seed=42)
sim = CardiacSimulator(config)
result = sim.run(population_preset("mi"))
result.to_csv("mi_population.csv")
```

CLI:

```bash
cardisim simulate --preset mi --cells 256 --days 28 --dt 0.25 --seed 42 --output mi.csv
cardisim derive-targets --expression expression.csv --metadata samples.json --dataset-id GSE185289 --study-id pig_regeneration --output targets.csv
```

## Inputs and outputs

**Inputs:** simulation configuration, phenotype presets or event parameters, population size, time step/duration, random seed, optional intervention parameters, and optional empirical data for calibration/target derivation.

**Outputs:** synthetic cell/phenotype trajectories, trajectory summaries, calibration/target tables, and CSV/JSON simulation artifacts with reproducibility metadata.

Synthetic values must not be represented as patient, animal, or cell measurements.

## Validation

Software tests cover simulator behavior and reproducibility. Calibration is incomplete until processed empirical data produce subject-level targets and fitted dynamics pass held-out validation. Empirical calibration should be assessed against data not used for fitting.

## Limitations

CardiSim is not a validated physiological model or digital twin. Default parameters are qualitative until externally calibrated. The state variables are abstractions and do not directly reproduce measured biomarkers. Synthetic trajectories cannot establish that a corresponding biological process will occur in vivo or in vitro.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.
