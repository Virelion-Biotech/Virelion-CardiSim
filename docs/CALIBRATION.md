# Empirical calibration

CardiSim can replace its hand-specified latent dynamics with parameters fitted from normalized empirical trajectories. The calibration target is a linear local approximation:

`dX/dt = b + A X + B F`

where `X` is the 12-dimensional latent phenotype vector and `F` is an optional observed forcing vector.

## CardiAtlas contract

CardiAtlas is the evidence/metadata source. A calibration manifest should preserve at least:

- Atlas dataset and study IDs
- public accession/source identifier
- organism
- modality
- condition
- time unit
- provenance URL

The checked-in CardiAtlas reference catalog is metadata-oriented; it does not itself constitute a numeric calibration set. CardiSim therefore requires normalized observations before fitting parameters.

## Observation format

The supported CSV is long-form with one row per cell/time/phenotype:

`dataset_id,study_id,subject_id,time,cell_id,phenotype,value`

All twelve phenotypes must be present for every cell and timepoint, and values must already be normalized to `[0,1]`. Optional columns `force_<phenotype>` encode a known forcing vector.

## Example

```python
from cardisim import calibration_from_csv, CardiacSimulator, SimulationConfig, population_preset

fit = calibration_from_csv("observations.csv", "dataset:gseXXXX", "study:gseXXXX")
print(fit.report.to_dict())
fit.save_json("calibration.json")

sim = CardiacSimulator(SimulationConfig(seed=42), dynamics=fit.parameters)
result = sim.run(population_preset("mi"))
print(result.summary()["dynamics_source"])
```

## Subject-disjoint validation

`calibrate_subject_holdout()` partitions complete biological subjects before fitting and evaluates one-step derivative error on held-out subjects. This is preferable to splitting individual technical rows because cell or replicate-level random splits can leak subject-specific dynamics.

```python
from cardisim import calibrate_subject_holdout, load_long_csv

data = load_long_csv("observations.csv", "dataset:gseXXXX", "study:gseXXXX")
checked = calibrate_subject_holdout(data, test_fraction=0.2, seed=42)
print(checked.report.to_dict())
```

## Parameter uncertainty

`bootstrap_calibrate()` resamples biological subjects with replacement and fits a parameter set for each bootstrap replicate. `BootstrapParameterEnsemble.summary()` reports quantiles for the intercept, state matrix, and forcing matrix, and `.simulate()` runs each parameter set under a common simulation configuration.

These are bootstrap sampling distributions. They are **not Bayesian posterior or credible intervals** unless a separate statistical model establishes that interpretation.

```python
from cardisim import bootstrap_calibrate

ensemble = bootstrap_calibrate(data, n_bootstrap=200, seed=42)
ensemble.save_json("dynamics_bootstrap.json")
runs = ensemble.simulate(SimulationConfig(seed=7, process_noise=0.0))
```

## Phenotype → CDT

After phenotype calibration, a separate paired dataset can fit the explicit phenotype→CDT profile. Keep the dynamics-calibration evidence and the phenotype→CDT evidence distinct; they answer different questions.

```text
empirical phenotype trajectories
          ↓
CardiSim dynamics calibration
          ↓
subject-disjoint validation
          ↓
phenotype state
          ↓
paired EP parameter observations
          ↓
phenotype→CDT profile
          ↓
CDT forward simulation
```

## Validation policy

Calibration and validation are separated. Parameters must be fit using a training subset defined by biological subject, not technical replicate, and checked against held-out subjects or independent studies in CardiEval. A high fit `R²` is not biological validation.

Numerical stability and timestep convergence are separate from biological calibration. CardiSim's validation helpers do not turn a stable numerical trajectory into evidence of physiological validity.

The simulator does not silently convert incomplete metadata into calibration evidence, and metadata-only CardiAtlas records cannot be used as numeric trajectories.
