import numpy as np

from cardisim import CardiacSimulator, SimulationConfig
from cardisim.models import PHENOTYPES
from cardisim.presets import population_preset, validate_presets


def test_reproducibility():
    cfg = SimulationConfig(duration=3, dt=0.25, n_cells=16, seed=123)
    a = CardiacSimulator(cfg).run(population_preset("mi"))
    b = CardiacSimulator(cfg).run(population_preset("mi"))
    np.testing.assert_array_equal(a.time, b.time)
    np.testing.assert_array_equal(a.values, b.values)


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


def test_presets_are_internally_complete():
    validate_presets()
    for name in ("baseline", "maturation", "mi", "hypoxia", "radiation", "electrotox"):
        assert population_preset(name) is not None
