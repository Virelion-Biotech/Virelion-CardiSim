# CardiSim model specification

## State-space model

For each synthetic cell, the state is a 12-dimensional vector `x(t)` with normalized values in `[0,1]`:

- productive dimensions: maturity, contractility, calcium handling, electrophysiology, metabolism, angiogenesis, viability, mitochondrial health
- burden dimensions: hypertrophy, fibrosis, inflammation, oxidative stress

The simulator uses the ordinary differential equation

`dx/dt = R * (h - x) + C * (x - h) + u(t)`

where `R` is a diagonal relaxation-rate matrix, `h` is a fixed homeostatic reference, `C` is a sparse qualitative coupling matrix, and `u(t)` is the sum of scheduled challenge forcing vectors.

The ODE is integrated using fourth-order Runge–Kutta. After each step, optional Gaussian process noise is applied and states are clipped to `[0,1]` by default.

## Event model

Each `ChallengeEvent` has:

- `onset`: start time
- `duration`: active interval
- `magnitude`: global amplitude
- `effects`: phenotype → signed forcing coefficient
- `recovery`: multiplicative attenuation factor

The default event envelope is `sin²(pi * phase)` across the event window. This gives zero forcing at onset and offset and maximal forcing at the event center.

## Population model

A population is generated from one canonical baseline state plus independent Gaussian heterogeneity. Cell IDs are deterministic (`cell_000000`, …) and all random operations use a NumPy generator seeded by `SimulationConfig.seed`.

## Presets

- `baseline`: no external perturbation.
- `maturation`: sustained positive maturation pressure.
- `mi`: acute myocardial-injury-like perturbation followed by a remodeling phase.
- `hypoxia`: short hypoxic stress.
- `radiation`: acute radiation-like injury followed by delayed remodeling.
- `electrotox`: acute electrophysiology/toxicity stress.

Preset names are intentionally descriptive and do **not** claim that the generated state trajectory quantitatively reproduces a real disease, dose response, cell type, animal, or patient.

## Extension points

The intended next-stage calibration path is:

1. Replace hand-set homeostasis/relaxation/coupling parameters with parameters fitted to `CardiAtlas` data.
2. Add explicit cell types and lineage-transition probabilities.
3. Fit challenge-specific event kernels from real time-course data.
4. Add uncertainty propagation and posterior parameter ensembles.
5. Emit standardized benchmark packages consumable by `CardiBench` and evaluated by `CardiEval`.
6. Connect learned surrogate models from `CardiLearn` while retaining the mechanistic baseline as a reference model.

## Validation philosophy

A simulation can be numerically stable and reproducible without being biologically valid. CardiSim therefore treats **numerical validation**, **software validation**, and **external biological calibration** as separate layers. Only the first two are provided by this repository's initial release.
