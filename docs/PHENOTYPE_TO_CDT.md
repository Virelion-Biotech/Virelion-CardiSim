# Phenotype → CDT calibration bridge

## Purpose

CardiSim is the latent phenotype-dynamics layer. CDT is the spatial electrophysiology layer. This bridge provides a machine-readable interface between them without embedding biological claims that have not been experimentally established.

## Default behavior

`uncalibrated_cdt_prior()` returns fixed CDT starting parameters with zero phenotype-dependent weights. This profile is explicitly marked `calibration_status = uncalibrated`.

It is valid for software integration and testing, but it must not be interpreted as evidence that fibrosis, inflammation, mitochondrial health, viability, or any other CardiSim variable changes conduction velocity or action-potential duration in a particular quantitative way.

## Calibrated behavior

`fit_phenotype_to_cdt()` fits a bounded affine mapping from the 12-dimensional normalized phenotype vector to:

- fibre conduction speed
- sheet conduction speed
- normal conduction speed
- dense endocardial speed
- sparse endocardial speed
- Purkinje speed
- minimum APD
- maximum APD

Phenotype values are centered at 0.5. Bounds prevent extrapolated parameters from silently leaving the declared search space.

## Data requirements

Paired samples should contain a phenotype matrix with the canonical CardiSim twelve variables and corresponding CDT parameter targets. The preferred validation design is subject-disjoint or study-disjoint rather than random splitting of technical replicates.

The fitted profile itself is not the final evidence. External predictive performance must be assessed on held-out paired data and, where available, independent cohorts.

## Recommended HeartTwin flow

```text
empirical observations
        ↓
CardiSim phenotype derivation / calibration
        ↓
subject-disjoint validation
        ↓
versioned phenotype→CDT profile
        ↓
HeartTwin
        ↓
CDT forward simulation
        ↓
CardiEval
```

## Serialization

Profiles are JSON-compatible and can be saved with `profile.save_json()` and reloaded with `load_profile()`. Keep the profile, fit report, source-data manifest, and held-out validation report together when claiming a calibrated mapping.
