import numpy as np
import pytest

from cardisim.calibration import EmpiricalTrajectory, calibrate
from cardisim.models import N_FEATURES


def test_calibration_recovers_linear_system():
    rng = np.random.default_rng(3)
    times = np.arange(0.0, 8.0, 0.25)
    n_cells = 8
    x0 = rng.uniform(0.35, 0.65, size=(n_cells, N_FEATURES))
    A = -0.015 * np.eye(N_FEATURES)
    b = 0.008 * np.ones(N_FEATURES)
    states = [x0]
    for _ in range(1, len(times)):
        x = states[-1]
        states.append(np.clip(x + 0.25 * (b + x @ A.T), 0.0, 1.0))
    data = EmpiricalTrajectory(
        "dataset-1", "study-1", times, np.stack(states),
        tuple(f"subject-{i}" for i in range(n_cells)),
        tuple(f"cell-{i}" for i in range(n_cells)),
    )
    result = calibrate(data, regularization=1e-10)
    assert result.report.rmse < 1e-7
    assert result.parameters.state_matrix.shape == (N_FEATURES, N_FEATURES)


def test_empirical_values_are_bounded():
    with pytest.raises(ValueError):
        EmpiricalTrajectory(
            "d", "s", np.array([0.0, 1.0]), np.ones((2, 1, N_FEATURES)) * 2,
            ("subject",), ("cell",),
        )
