"""Deterministic synthetic cardiac-regeneration/genotype challenge trajectories.

All axes are normalized latent challenge variables, not biomarkers, fitted biology,
patient predictions, or expected effects of named therapies.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
from typing import Any
import numpy as np

SCENARIO_VERSION="1.0.0"
GENERATOR_VERSION="cardisim-regenerative-scenarios/1.0.0"
SCENARIOS={
 "post_mi_transient_proliferation":"Post-MI injury with a transient MYC–Cyclin-T1-like proliferation pulse and independent safety/remodelling axes",
 "pln_r14del_calcium":"PLN-R14del-like calcium-handling challenge with a synthetic target-modulation input",
 "bag3_dcm_proteostasis":"BAG3-associated DCM-like proteostasis and contractility challenge",
 "pkp2_arrhythmogenic":"PKP2-associated arrhythmogenic-cardiomyopathy-like desmosomal and electrical challenge",
 "lamp2_danon_autophagy":"LAMP2/Danon-like autophagic-lysosomal dysfunction and hypertrophic-remodelling challenge",
 "immune_resolution_ifng":"Synthetic IFN-gamma/macrophage/matrix-remodelling resolution challenge"
}

@dataclass(frozen=True)
class RegenerativeScenarioResult:
    scenario_id:str
    time:np.ndarray
    axes:dict[str,np.ndarray]
    manifest:dict[str,Any]
    def mean_trajectory(self)->dict[str,list[float]]:
        return {k:v.mean(axis=0).tolist() for k,v in self.axes.items()}
    def fingerprint(self)->str:
        h=hashlib.sha256(json.dumps(self.manifest,sort_keys=True,separators=(",",":")).encode())
        h.update(np.ascontiguousarray(self.time).tobytes())
        for key in sorted(self.axes): h.update(key.encode()); h.update(np.ascontiguousarray(self.axes[key]).tobytes())
        return h.hexdigest()
    def to_dict(self)->dict[str,Any]:
        return {"scenario_id":self.scenario_id,"time_days":self.time.tolist(),"axes":{k:v.tolist() for k,v in self.axes.items()},"manifest":self.manifest,"fingerprint":self.fingerprint()}

def _grid(duration:float,dt:float)->np.ndarray:
    if not np.isfinite(duration) or not np.isfinite(dt) or duration<=0 or dt<=0 or duration<dt:
        raise ValueError("duration and dt must be finite, positive, and duration >= dt")
    n=int(np.floor(duration/dt+1e-12)); t=np.arange(n+1,dtype=float)*dt
    if t[-1]<duration-1e-10: t=np.append(t,duration)
    else: t[-1]=duration
    return t

def _sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-40,40)))
def _clip(x): return np.clip(x,0.0,1.0)

def simulate_regenerative_scenario(scenario_id:str,*,duration:float=28.0,dt:float=0.25,seed:int=7,n_samples:int=32,heterogeneity:float=0.015)->RegenerativeScenarioResult:
    """Generate a seeded cohort of bounded, explicitly synthetic latent trajectories."""
    key=scenario_id.strip().lower()
    if key not in SCENARIOS: raise KeyError(f"unknown scenario {scenario_id!r}; choose from {sorted(SCENARIOS)}")
    if not isinstance(n_samples,int) or n_samples<1: raise ValueError("n_samples must be a positive integer")
    if not np.isfinite(heterogeneity) or heterogeneity<0: raise ValueError("heterogeneity must be finite and non-negative")
    t=_grid(float(duration),float(dt)); rng=np.random.default_rng(seed)
    # Static per-sample offsets preserve seeded reproducibility without timestep-dependent noise.
    offsets=lambda scale=1.0: rng.normal(0.0,heterogeneity*scale,size=(n_samples,1))
    tt=t[None,:]
    axes={}
    if key=="post_mi_transient_proliferation":
        pulse=np.exp(-0.5*((tt-4.0)/1.5)**2)
        cumulative=np.cumsum(np.r_[0.0,np.diff(t)]*(pulse[0,:])) if len(t)>1 else np.zeros(1)
        cumulative=cumulative/max(float(cumulative[-1]),1e-12)
        axes={
         "cell_cycle_entry":_clip(0.08+0.78*pulse+offsets()),
         "cell_number_proxy":_clip(0.48+0.13*cumulative[None,:]+offsets(0.5)),
         "contractility_latent":_clip(0.38+0.18*(1-np.exp(-tt/18))+offsets()),
         "fibrosis_latent":_clip(0.52-0.07*(1-np.exp(-tt/35))+offsets()),
         "arrhythmia_risk_latent":_clip(0.16+0.58*pulse+offsets()),
         "off_target_proliferation_latent":_clip(0.06+0.42*pulse+offsets())
        }
    elif key=="pln_r14del_calcium":
        modulation=_sigmoid((tt-7.0)/1.8)
        axes={"calcium_handling_latent":_clip(0.28+0.35*modulation+offsets()),"target_modulation_latent":_clip(0.05+0.72*modulation+offsets()),"arrhythmia_risk_latent":_clip(0.68-0.30*modulation+offsets()),"contractility_latent":_clip(0.42+0.10*modulation+offsets())}
    elif key=="bag3_dcm_proteostasis":
        decline=1-np.exp(-tt/35)
        axes={"proteostasis_latent":_clip(0.62-0.35*decline+offsets()),"sarcomeric_integrity_latent":_clip(0.66-0.28*decline+offsets()),"contractility_latent":_clip(0.62-0.30*decline+offsets()),"hypertrophy_latent":_clip(0.25+0.42*decline+offsets())}
    elif key=="pkp2_arrhythmogenic":
        progression=tt/max(duration,1e-12)
        axes={"desmosomal_integrity_latent":_clip(0.72-0.42*progression+offsets()),"electrophysiology_instability_latent":_clip(0.18+0.56*progression+offsets()),"arrhythmia_risk_latent":_clip(0.20+0.62*progression+offsets()),"fibrosis_latent":_clip(0.12+0.30*progression+offsets())}
    elif key=="lamp2_danon_autophagy":
        impairment=1-np.exp(-tt/24)
        axes={"autophagic_flux_latent":_clip(0.68-0.42*impairment+offsets()),"lysosomal_function_latent":_clip(0.70-0.40*impairment+offsets()),"hypertrophy_latent":_clip(0.22+0.48*impairment+offsets()),"contractility_latent":_clip(0.62-0.16*impairment+offsets())}
    else:
        ifng=0.72*np.exp(-tt/9)+0.08
        macrophage=0.18+0.58*np.exp(-((tt-4)/8)**2)
        matrix=0.62-0.25*(1-np.exp(-tt/20))
        axes={"ifng_signalling_latent":_clip(ifng+offsets()),"macrophage_activity_latent":_clip(macrophage+offsets()),"matrix_remodelling_latent":_clip(matrix+offsets()),"fibrosis_latent":_clip(0.55-0.16*(1-np.exp(-tt/22))+offsets())}
    for name,values in axes.items():
        if values.shape!=(n_samples,len(t)) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
            raise FloatingPointError(f"scenario axis {name} violated finite/bounded output contract")
    manifest={"schema_version":"1.0","scenario_version":SCENARIO_VERSION,"generator_version":GENERATOR_VERSION,"scenario_id":key,"description":SCENARIOS[key],"evidence_state":"synthetic","seed":int(seed),"n_samples":n_samples,"duration_days":float(duration),"dt_days":float(dt),"heterogeneity_sd":float(heterogeneity),"axes":sorted(axes),"axis_interpretation":"normalized latent challenge variables; not measured biomarkers","calibration_status":"uncalibrated_synthetic","clinical_or_therapeutic_efficacy_inference":False}
    return RegenerativeScenarioResult(key,t,axes,manifest)

def timestep_sensitivity(scenario_id:str,*,duration:float=28.0,dt:float=0.5,seed:int=7,n_samples:int=32)->dict[str,Any]:
    """Compare mean latent trajectories at dt and dt/2 using linear interpolation."""
    coarse=simulate_regenerative_scenario(scenario_id,duration=duration,dt=dt,seed=seed,n_samples=n_samples,heterogeneity=0.0)
    fine=simulate_regenerative_scenario(scenario_id,duration=duration,dt=dt/2,seed=seed,n_samples=n_samples,heterogeneity=0.0)
    errors={}
    for key in coarse.axes:
        interp=np.interp(coarse.time,fine.time,fine.axes[key].mean(axis=0))
        errors[key]=float(np.sqrt(np.mean((coarse.axes[key].mean(axis=0)-interp)**2)))
    return {"scenario_id":scenario_id,"dt_coarse":dt,"dt_fine":dt/2,"rmse_by_axis":errors,"max_rmse":max(errors.values(),default=0.0),"interpretation":"numerical discretization comparison only; not biological validation"}
