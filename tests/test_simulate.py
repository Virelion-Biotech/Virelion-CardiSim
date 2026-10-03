import numpy as np

from cardisim import CardiacSimulator, SimulationConfig, uncalibrated_cdt_prior
from cardisim.events import ChallengeEvent, EventSchedule
from cardisim.models import PHENOTYPES
from cardisim.presets import population_preset, validate_presets


def test_reproducibility():
    cfg = SimulationConfig(duration=3, dt=0.25, n_cells=16, seed=123)
    a = CardiacSimulator(cfg).run(population_preset("mi"))
    b = CardiacSimulator(cfg).run(population_preset("mi"))
    np.testing.assert_array_equal(a.time, b.time)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.fingerprint() == b.fingerprint()
    assert a.reproducibility_manifest() == b.reproducibility_manifest()


def test_manifest_preserves_event_definitions_and_dynamics_identity():
    schedule = EventSchedule((ChallengeEvent("test", onset=0.0, duration=1.0, kernel="box", effects={"fibrosis": 0.2}),))
    result = CardiacSimulator(
        SimulationConfig(duration=1.0, dt=0.5, n_cells=4, seed=4, process_noise=0.0)
    ).run(schedule)
    manifest = result.reproducibility_manifest()
    assert manifest["event_specs"][0]["kernel"] == "box"
    assert manifest["initial_state_fingerprint"]
    assert manifest["dynamics_fingerprint"] == "default"
    assert manifest["fingerprint"] == result.fingerprint()


def test_population_shape_and_bounds():
    cfg = SimulationConfig(duration=2, dt=0.2, n_cells=11, seed=2)
    result = CardiacSimulator(cfg).run(population_preset("hypoxia"))
    assert result.values.shape == (len(result.time), 11, len(PHENOTYPES))
    assert np.all(result.values >= 0)
    assert np.all(result.values <= 1)


def test_mi_changes_health_relevant_states():
    cfg = SimulationConfig(duration=5, dt=0.25, n_cells=64, seed=4, process_noise=0)
    baseline = CardiacSimulator(cfg).run(population_preset("baseline"))
    injury = CardiacSimulator(cfg).run(population_preset("mi"))
    assert injury.final.mean()["viability"] < baseline.final.mean()["viability"]
    assert injury.final.mean()["fibrosis"] > baseline.final.mean()["fibrosis"]


def test_custom_initial_mean_is_used():
    cfg = SimulationConfig(duration=0.5, dt=0.25, n_cells=8, seed=2, heterogeneity=0.0, process_noise=0.0)
    initial = {name: 0.5 for name in PHENOTYPES}
    initial["fibrosis"] = 0.9
    result = CardiacSimulator(cfg).run(initial=initial)
    assert result.initial.mean()["fibrosis"] == 0.9
    assert result.initial.mean()["maturity"] == 0.5
    assert result.initialization == "custom-mean+heterogeneity"


def test_cdt_export_uses_explicit_profile():
    cfg = SimulationConfig(duration=0.5, dt=0.25, n_cells=4, seed=3, heterogeneity=0.0, process_noise=0.0)
    result = CardiacSimulator(cfg).run()
    params = result.cdt_parameters(uncalibrated_cdt_prior())
    assert params["fibre_speed"] == 0.065
    assert params["apd_max"] >= params["apd_min"]


def test_presets_are_internally_complete():
    validate_presets()
    for name in ("baseline", "maturation", "mi", "hypoxia", "radiation", "electrotox"):
        assert population_preset(name) is not None
