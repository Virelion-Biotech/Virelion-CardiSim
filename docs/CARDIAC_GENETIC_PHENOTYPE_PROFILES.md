# Cardiac genetic phenotype profiles

This document defines a staged, low-compute interface for representing genotype-linked
cardiac mechanisms. It does not claim that the repository currently simulates the
full molecular or electrophysiological consequences of these variants.

## CASQ2 / RYR2: calcium handling and CPVT

Represent the mechanistic chain as separately evidenced components:

1. Sarcoplasmic-reticulum calcium storage/release regulation.
2. Diastolic calcium leak or abnormal release propensity (when supported by the
   specific model or source).
3. Cytosolic calcium transient and calcium removal kinetics.
4. Triggered activity and delayed afterdepolarisations.
5. Exercise/catecholamine context and ventricular arrhythmia phenotype.

Parameters must be model-specific, sourced and accompanied by units and uncertainty.
Do not assign a universal parameter change to every CASQ2 or RYR2 variant.

## TNNT2: contractile phenotype

Represent sarcomeric/contractile effects independently from electrical conduction.
Where supported by data, track force generation, calcium sensitivity, relaxation,
sarcomere organisation and downstream remodelling as distinct variables.

## Minimal implementation path

- Add versioned phenotype-profile JSON inputs with provenance and explicit missingness.
- Use existing CPU-compatible model components for sensitivity analyses.
- Compare outputs against published, source-adjudicated measurements before calling
  any profile validated.
- Report parameter identifiability and uncertainty; do not overfit a single phenotype.
- Keep preclinical mechanistic plausibility separate from clinical efficacy.

## Interoperability

Profiles should link to CardiAtlas genetic-target/product evidence and ElectroTrace
measurement bundles through stable IDs and schema versions. The first deliverable
is a transparent metadata bridge, not a new high-cost multiscale solver.
