import numpy as np

from cardisim import CardiacSimulator, EmpiricalTrajectory, SimulationConfig, bootstrap_calibrate
from cardisim.models import N_FEATURES


def make_data():
    rng = np.random.default_rng(21)
    times = np.arange(0.0, 4.0, 1.0)
    values = []
    subjects = []
    cells = []
    for i in range(6):
        x = rng.uniform(0.35, 0.65, size=N_FEATURES)
        rows = [x.copy()]
        for _ in range(1, len(times)):
            x = np.clip(x + 0.01 * (0.5 - x), 0, 1)
            rows.append(x.copy())
        values.append(np.stack(rows))
        subjects.append(f"subject-{i}")
        cells.append(f"cell-{i}")
    return EmpiricalTrajectory("d", "s", times, np.stack(values), tuple(subjects), tuple(cells))


def test_bootstrap_is_reproducible_and_simulatable():
    data = make_data()
    a = bootstrap_calibrate(data, n_bootstrap=4, seed=7, regularization=1e-4)
    b = bootstrap_calibrate(data, n_bootstrap=4, seed=7, regularization=1e-4)
    assert len(a.parameters) == 4
    np.testing.assert_array_equal(a.parameters[0].state_matrix, b.parameters[0].state_matrix)
    summary = a.summary()
    assert summary["n_subjects"] == 6
    assert len(summary["state_matrix"]["q50"]) == N_FEATURES
    runs = a.simulate(SimulationConfig(duration=1.0, dt=0.5, n_cells=4, seed=2, process_noise=0.0))
    assert len(runs) == 4
    assert all(np.isfinite(run.values).all() for run in runs)
