import numpy as np
import pytest

from cardisim.regenerative_scenarios import (
    SCENARIOS,
    simulate_regenerative_scenario,
    timestep_sensitivity,
)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scenarios_are_seeded_finite_and_bounded(scenario):
    first = simulate_regenerative_scenario(
        scenario,
        duration=14,
        dt=0.25,
        seed=12,
        n_samples=10,
    )
    second = simulate_regenerative_scenario(
        scenario,
        duration=14,
        dt=0.25,
        seed=12,
        n_samples=10,
    )
    assert first.fingerprint() == second.fingerprint()
    assert first.manifest["evidence_state"] == "synthetic"
    assert all(
        np.isfinite(values).all()
        and ((values >= 0) & (values <= 1)).all()
        for values in first.axes.values()
    )


def test_post_mi_pulse_separates_proliferation_from_risk_and_contractility():
    result = simulate_regenerative_scenario(
        "post_mi_transient_proliferation",
        duration=20,
        dt=0.1,
        seed=1,
        n_samples=1,
        heterogeneity=0,
    )
    proliferation = result.axes["cell_cycle_entry"][0]
    arrhythmia_risk = result.axes["arrhythmia_risk_latent"][0]
    assert proliferation.max() > proliferation[0]
    assert arrhythmia_risk.max() > arrhythmia_risk[0]
    assert "fibrosis_latent" in result.axes
    assert "contractility_latent" in result.axes
    assert not np.array_equal(proliferation, arrhythmia_risk)


def test_pln_modulation_axis_and_calcium_handling_are_separate():
    result = simulate_regenerative_scenario(
        "pln_r14del_calcium",
        duration=20,
        dt=0.25,
        seed=2,
        n_samples=1,
        heterogeneity=0,
    )
    target_modulation = result.axes["target_modulation_latent"][0]
    calcium_handling = result.axes["calcium_handling_latent"][0]
    assert target_modulation[-1] > target_modulation[0]
    assert calcium_handling[-1] > calcium_handling[0]


def test_unknown_scenario_and_invalid_grid_fail():
    with pytest.raises(KeyError):
        simulate_regenerative_scenario("unvalidated_therapy")
    with pytest.raises(ValueError):
        simulate_regenerative_scenario(
            "pkp2_arrhythmogenic",
            dt=0,
        )


def test_timestep_sensitivity_reports_numerical_not_biological_metric():
    report = timestep_sensitivity(
        "lamp2_danon_autophagy",
        duration=14,
        dt=0.5,
    )
    assert np.isfinite(report["max_rmse"])
    assert report["max_rmse"] < 0.01
    assert "not biological validation" in report["interpretation"]
