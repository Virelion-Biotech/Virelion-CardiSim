# Synthetic regenerative and genotype-specific challenge scenarios

CardiSim adds six seeded synthetic challenge generators. They are qualitative software test fixtures, not fitted biological models, patient predictions, biomarkers, or expected effects of named therapies.

## Presets

- `post_mi_transient_proliferation`: transient cell-cycle-entry pulse with separate cell-number proxy, contractility, fibrosis, arrhythmia-risk and off-target-proliferation axes.
- `pln_r14del_calcium`: calcium-handling impairment and a synthetic target-modulation input.
- `bag3_dcm_proteostasis`: proteostasis/sarcomeric-integrity decline, contractility and hypertrophy.
- `pkp2_arrhythmogenic`: desmosomal-integrity decline and separate electrical-instability/arrhythmia axes.
- `lamp2_danon_autophagy`: autophagic/lysosomal dysfunction and hypertrophic-remodelling axes.
- `immune_resolution_ifng`: IFN-gamma, macrophage activity and matrix-remodelling trajectories.

## Usage

```python
from cardisim.regenerative_scenarios import simulate_regenerative_scenario, timestep_sensitivity

result = simulate_regenerative_scenario(
    "post_mi_transient_proliferation",
    duration=28,
    dt=0.25,
    seed=42,
    n_samples=64,
)
result.to_dict()  # includes synthetic label, generator version, seed and fingerprint
print(timestep_sensitivity("post_mi_transient_proliferation", dt=0.5))
```

The existing twelve-dimensional canonical phenotype state is unchanged. New mechanism-specific axes are sidecar latent variables so they cannot silently be mistaken for measured biomarkers or be passed into the phenotype-to-CDT mapping. The post-MI challenge explicitly allows proliferation to rise while arrhythmia/off-target-risk axes also rise; contractility and fibrosis are separate and are not inferred from cell-cycle entry.

All axes are bounded to [0,1], deterministic for a fixed seed/configuration, and explicitly marked `evidence_state=synthetic` and `calibration_status=uncalibrated_synthetic`. Timestep sensitivity compares numerical interpolation only. No treatment efficacy or clinical inference is provided.
