# CardiSim model specification

## Role in HeartTwin

CardiSim is the **phenotype-dynamics layer**. It represents longitudinal latent cardiac state, controlled perturbations, population variability, and empirically calibrated phenotype dynamics. Spatial electrophysiology remains a separate CDT layer consumed by HeartTwin.

## State-space model

For each synthetic cell, the state is a 12-dimensional vector `x(t)` with normalized values in `[0,1]`:

- productive dimensions: maturity, contractility, calcium handling, electrophysiology, metabolism, angiogenesis, viability, mitochondrial health
- burden dimensions: hypertrophy, fibrosis, inflammation, oxidative stress

The simulator uses the ordinary differential equation

`dx/dt = b + A x + B u(t)`

where `b`, `A`, and optional forcing matrix `B` are either qualitative defaults or empirically fitted parameters. The packaged default dynamics are a bounded qualitative prior, not a physiological law.

The ODE is integrated using fourth-order Runge–Kutta. After each step, optional Gaussian process noise is applied and states are clipped to `[0,1]` by default.

## Event model

Each `ChallengeEvent` has:

- `onset`: start time
- `duration`: active interval
- `magnitude`: global amplitude
- `effects`: phenotype → signed forcing coefficient
- `recovery`: multiplicative attenuation factor

The default event envelope is `sin²(pi * phase)` across the event window. Multiple events can overlap and their forcing vectors are summed.

## Population model

A population is generated from a canonical baseline or an explicitly supplied phenotype mean plus independent Gaussian heterogeneity. A validated `CardiacState` can also be provided directly, which enables HeartTwin to seed a subsequent longitudinal simulation from an existing phenotype state.

Cell IDs are deterministic unless a supplied state includes its own IDs. Random operations use a NumPy generator seeded by `SimulationConfig.seed`.

## Empirical calibration

`calibrate()` fits the local linear dynamics to processed normalized trajectories. `calibrate_subject_holdout()` additionally performs a subject-disjoint fit/evaluation split so technical replicates do not masquerade as independent biological validation.

The preferred evidence path is:

```text
CardiAtlas metadata
      ↓
processed subject-level phenotype targets
      ↓
CardiSim calibration
      ↓
subject-disjoint validation
      ↓
independent study validation
```

An in-sample fit statistic is not external biological validation.

## Phenotype → CDT bridge

`phenotype_to_cdt.py` defines the explicit interface from latent phenotype means to CDT parameters. The default profile is uncalibrated and phenotype-independent. `fit_phenotype_to_cdt()` can fit phenotype-dependent affine rules from paired phenotype/EP parameter observations; these profiles are versioned, bounded, serializable, and separately testable.

The intended chain is:

```text
CardiSim phenotype state
       ↓
calibrated phenotype→CDT profile
       ↓
CDT parameter prior / target
       ↓
spatial electrophysiology simulation
```

A fitted mapping must be evaluated on held-out paired data before biological interpretation.

## Numerical validation

`dynamics_stability()` reports the spectral abscissa of the continuous-time linear state Jacobian. `timestep_convergence()` runs deterministic coarse/refined simulations with process noise and state clipping disabled so numerical error is measured independently of stochastic/clamping behavior. `diagnose_result()` checks finiteness, monotonic time, bounds, and boundary occupancy.

## Uncertainty language

`state_spread()` reports population variability across simulated cells. `ensemble_final_means()` reports run-to-run variability across independent simulations. Neither is a confidence interval or posterior credible interval by itself.

## Neural surrogate layer

CardiGNN and CardiGINO are acceleration/benchmark components. Their evaluation must use geometry-disjoint splits where repeated geometries are grouped, because random case splits can overstate generalization.

## Presets

- `baseline`: no external perturbation.
- `maturation`: sustained positive maturation pressure.
- `mi`: acute myocardial-injury-like perturbation followed by a remodeling phase.
- `hypoxia`: short hypoxic stress.
- `radiation`: acute radiation-like injury followed by delayed remodeling.
- `electrotox`: acute electrophysiology/toxicity stress.

Preset names are intentionally descriptive and do not claim that the generated trajectory quantitatively reproduces a real disease, dose response, cell type, animal, or patient.

## Validation philosophy

CardiSim separates **software validation**, **numerical validation**, and **biological calibration/validation**. The repository can establish reproducibility and numerical behavior; external empirical calibration is required before treating parameters or phenotype-to-CDT mappings as biological relationships.
