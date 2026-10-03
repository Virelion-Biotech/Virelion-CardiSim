# Empirical calibration panel

CardiSim is calibrated in layers. The repository stores a public accession manifest, not redistributed raw/large GEO matrices.

## Locked panel

| Accession | Organism | Modality | Primary role | Key structure |
|---|---|---|---|---|
| GSE185289 | pig | snRNA-seq | maturation + regeneration + remodeling | fetal/postnatal controls; ARP1+MI regenerative arm; MI-only non-regenerative arm |
| GSE240848 | rat | snRNA-seq | acute MI / ischemia-reperfusion | sham, MI 1 h, 6 h, 24 h; infarct-zone nuclei |
| GSE135310 | mouse | scRNA-seq | post-MI inflammation | steady state, sham, and post-MI days 1, 3, 5 inflammatory populations |

## Why these three

GSE185289 is the anchor because it spans developmental state and a mechanistically useful regenerative-versus-non-regenerative injury comparison in a large-animal model. Its published design explicitly includes fetal and postnatal controls plus ARP1, MI-only, and combined ARP1+MI groups across multiple harvest times.

GSE240848 supplies a compact acute-injury trajectory with sham and ischemia/reperfusion controls, allowing the fast injury component of the simulator to be challenged separately from long remodeling.

GSE135310 supplies a time-resolved immune component that should prevent CardiSim from treating inflammation as a generic scalar burden with no temporal structure.

## Data policy

The panel is intentionally metadata-only in Git. CardiSim should fetch processed public expression files at calibration time, record the exact source URL/file checksum, and write derived subject-level phenotype targets to a versioned calibration artifact.

Do **not** fit parameters from individual cells treated as independent biological replicates. The fitting unit should be the biological sample/subject, with technical replicates grouped before parameter estimation.

## Target derivation

The first calibration target is a 12-dimensional latent phenotype vector. These dimensions are not direct measurements. They are derived from prespecified marker/module sets and normalized within a dataset:

- maturity
- contractility
- calcium_handling
- electrophysiology
- metabolism
- hypertrophy
- fibrosis
- inflammation
- angiogenesis
- viability
- oxidative_stress
- mitochondrial_health

Each derived target must retain:

- dataset accession
- sample accession
- biological subject identifier when available
- condition
- timepoint
- cell context
- module/marker set version
- normalization method
- source-file checksum

## Calibration gate

A dataset is eligible for parameter fitting only when:

1. required phenotypes are measurable or explicitly marked missing;
2. at least two biological timepoints exist for temporal fitting;
3. biological subject IDs can be grouped, or the dataset is explicitly classified as aggregate-only;
4. no test sample contributes to parameter selection;
5. raw/processed source provenance is complete;
6. the resulting fit passes held-out validation.

Until those conditions are met, the default hand-specified dynamics remain the reference model.
