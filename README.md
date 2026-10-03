# Virelion-CardiSim

CardiSim is the Virelion **phenotype-dynamics engine**. It generates reproducible synthetic cardiac-cell and cardiac-phenotype trajectories under controlled perturbations, fits latent dynamics to empirical trajectories, summarizes population variability, and provides an explicit calibration-ready bridge into a spatial cardiac digital-twin engine.

CardiSim is not a validated physiological model or clinical digital twin. Its normalized phenotype variables are latent abstractions unless an external calibration study establishes a measurement relationship.

## What it contains

- Bounded twelve-dimensional phenotype dynamics.
- Fourth-order Runge–Kutta integration.
- Seeded population heterogeneity and deterministic cell IDs.
- Time-localized challenge and intervention events.
- Baseline, maturation, myocardial-injury-like, hypoxia, radiation-like, and electrophysiology/toxicity presets.
- Empirical trajectory, longitudinal, cohort, and subject-disjoint calibration utilities.
- Numerical stability and timestep-convergence diagnostics.
- Population-spread and run-to-run variability summaries.
- Explicit, versioned phenotype→CDT parameter mapping profiles.
- Geometry-aware CardiGNN and geometry-informed CardiGINO research models.
- Geometry-disjoint benchmarking for neural surrogates.
- CardiAtlas metadata bridges and auditable target derivation.
- Python API and CLI.

## Canonical phenotype state

The state dimensions are `maturity`, `contractility`, `calcium_handling`, `electrophysiology`, `metabolism`, `hypertrophy`, `fibrosis`, `inflammation`, `angiogenesis`, `viability`, `oxidative_stress`, and `mitochondrial_health`. They are normalized latent variables, not direct patient, animal, or cellular measurements.

## Installation

```bash
pip install -e .
pip install -e '.[dev]'
```

## Usage

Basic simulation:

```python
from cardisim import CardiacSimulator, SimulationConfig, population_preset

config = SimulationConfig(duration=28, dt=0.25, n_cells=256, seed=42)
result = CardiacSimulator(config).run(population_preset("mi"))
result.to_csv("mi_population.csv")
print(result.fingerprint())
```

Start from a supplied phenotype mean and export CDT parameters:

```bash
cardisim simulate \
  --preset mi \
  --cells 256 \
  --days 28 \
  --dt 0.25 \
  --seed 42 \
  --initial-json patient_phenotype.json \
  --output mi_population.csv \
  --cdt-output cdt_parameters.json
```

Use a calibrated phenotype→CDT profile:

```bash
cardisim simulate \
  --preset baseline \
  --initial-json phenotype.json \
  --cdt-profile profiles/phenotype_to_cdt_v1.json \
  --cdt-output cdt_parameters.json \
  --output trajectory.json \
  --format json
```

Derive calibration targets from an expression table:

```bash
cardisim derive-targets \
  --expression expression.csv \
  --metadata samples.json \
  --dataset-id GSE185289 \
  --study-id pig_regeneration \
  --output targets.csv
```

## Phenotype → CDT bridge

`cardisim.phenotype_to_cdt` provides a strict handoff contract from the latent phenotype layer to CDT electrophysiology parameters. The packaged profile is intentionally uncalibrated and has zero phenotype-dependent weights. This prevents the software from silently asserting disease-to-conduction relationships that have not been established.

Paired empirical data can be used to fit affine mappings for fibre, sheet, normal, endocardial and Purkinje conduction speeds and APD bounds. The fitted profile is versioned and serializable, and held-out paired data can be evaluated separately.

```python
from cardisim import fit_phenotype_to_cdt, validate_phenotype_to_cdt

profile, fit_report = fit_phenotype_to_cdt(phenotype_matrix, cdt_parameter_targets)
profile.save_json("phenotype_to_cdt_v1.json")
held_out_report = validate_phenotype_to_cdt(profile, held_out_phenotypes, held_out_targets)
```

## Validation layers

CardiSim separates three claims:

1. **Software validation:** deterministic behavior, schema/API checks, bounded-state checks, unit tests, and reproducibility fingerprints.
2. **Numerical validation:** linear-system stability diagnostics and timestep-refinement comparisons with stochastic noise/clipping disabled.
3. **Biological calibration/validation:** parameters fitted from processed empirical data and evaluated on held-out subjects or independent datasets.

A low numerical error does not establish biological validity. A high calibration R² does not establish external predictive validity.

## Integration with HeartTwin

CardiSim remains the phenotype layer. HeartTwin can consume its simulation summaries and, when a calibrated mapping profile is supplied, translate a final phenotype state into CDT parameter targets. The mapping artifact should retain its calibration provenance and validation results.

## Limitations

Default phenotype equations and presets are qualitative. The simulator does not reproduce measured biomarkers by construction, and synthetic trajectories cannot establish that a corresponding biological process occurs in vivo or in vitro. The phenotype→CDT bridge is a calibration interface; the default fixed prior is not a validated biological mapping.

## License

GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See `LICENSE`.
