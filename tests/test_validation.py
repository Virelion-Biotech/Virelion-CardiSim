import numpy as np

from cardisim import CardiacSimulator, SimulationConfig, population_preset
from cardisim.dynamics import DEFAULT_PARAMETERS
from cardisim.uncertainty import ensemble_final_means, state_spread
from cardisim.validation import diagnose_result, dynamics_stability, timestep_convergence


def test_default_dynamics_are_stable():
    report = dynamics_stability(DEFAULT_PARAMETERS)
    assert report.eigenvalues_real_max < 0
    assert report.stable


def test_result_diagnostics_and_population_spread():
    result = CardiacSimulator(
        SimulationConfig(duration=2.0, dt=0.25, n_cells=16, seed=8, process_noise=0.0)
    ).run(population_preset("mi"))
    diagnostics = diagnose_result(result)
    assert diagnostics.finite
    assert diagnostics.monotonic_time
    assert diagnostics.bounded
    spread = state_spread(result.final)
    assert spread.n == 16
    assert spread.q025["viability"] <= spread.median["viability"] <= spread.q975["viability"]
    ensemble = ensemble_final_means([result, result])
    assert ensemble.n == 2


def test_rk4_timestep_refinement_is_deterministic():
    simulator = CardiacSimulator(
        SimulationConfig(duration=2.0, dt=0.5, n_cells=8, seed=11, process_noise=0.0)
    )
    report = timestep_convergence(simulator, population_preset("maturation"), refinement_factor=2, tolerance=1e-8)
    assert report.refined_dt == 0.25
    assert np.isfinite(report.trajectory_rmse)
    assert report.trajectory_rmse < 1e-4
