import numpy as np
import pytest
from cardisim.regenerative_scenarios import SCENARIOS, simulate_regenerative_scenario, timestep_sensitivity

@pytest.mark.parametrize("scenario",list(SCENARIOS))
def test_scenarios_are_seeded_finite_and_bounded(scenario):
    a=simulate_regenerative_scenario(scenario,duration=14,dt=0.25,seed=12,n_samples=10)
    b=simulate_regenerative_scenario(scenario,duration=14,dt=0.25,seed=12,n_samples=10)
    assert a.fingerprint()==b.fingerprint()
    assert a.manifest["evidence_state"]=="synthetic"
    assert all(np.isfinite(x).all() and ((x>=0)&(x<=1)).all() for x in a.axes.values())

def test_post_mi_pulse_separates_proliferation_from_risk_and_contractility():
    r=simulate_regenerative_scenario("post_mi_transient_proliferation",duration=20,dt=0.1,seed=1,n_samples=1,heterogeneity=0)
    p=r.axes["cell_cycle_entry"][0]
    risk=r.axes["arrhythmia_risk_latent"][0]
    assert p.max()>p[0] and risk.max()>risk[0]
    assert "fibrosis_latent" in r.axes and "contractility_latent" in r.axes
    assert not np.array_equal(p,risk)

def test_pln_modulation_axis_and_calcium_handling_are_separate():
    r=simulate_regenerative_scenario("pln_r14del_calcium",duration=20,dt=0.25,seed=2,n_samples=1,heterogeneity=0)
    assert r.axes["target_modulation_latent"][0,-1]>r.axes["target_modulation_latent"][0,0]
    assert r.axes["calcium_handling_latent"][0,-1]>r.axes["calcium_handling_latent"][0,0]

def test_unknown_scenario_and_invalid_grid_fail():
    with pytest.raises(KeyError): simulate_regenerative_scenario("unvalidated_therapy")
    with pytest.raises(ValueError): simulate_regenerative_scenario("pkp2_arrhythmogenic",dt=0)

def test_timestep_sensitivity_reports_numerical_not_biological_metric():
    report=timestep_sensitivity("lamp2_danon_autophagy",duration=14,dt=0.5)
    assert np.isfinite(report["max_rmse"]) and report["max_rmse"]<0.01
    assert "not biological validation" in report["interpretation"]
