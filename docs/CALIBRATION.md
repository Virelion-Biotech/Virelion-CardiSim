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

## Validation policy

Calibration and validation are separated. Parameters must be fit using a training subset defined by biological subject, not technical replicate, and checked against held-out subjects or independent studies in CardiEval. A high fit `R²` is not biological validation.

The simulator does not silently convert incomplete metadata into calibration evidence, and metadata-only CardiAtlas records cannot be used as numeric trajectories.
