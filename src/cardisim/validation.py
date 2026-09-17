"""Numerical validation helpers for the CardiSim phenotype engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from .dynamics import DynamicsParameters
from .events import EventSchedule
from .models import SimulationConfig


@dataclass(frozen=True)
class DynamicsStabilityReport:
    """Continuous-time linear stability diagnostics for the state Jacobian."""

    eigenvalues_real_max: float
    eigenvalues_real_min: float
    stable: bool
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dynamics_stability(
    parameters: DynamicsParameters,
    tolerance: float = 1e-10,
) -> DynamicsStabilityReport:
    """Report the spectral abscissa of the calibrated linear dynamics."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    eigenvalues = np.linalg.eigvals(np.asarray(parameters.state_matrix, dtype=float))
    real = np.real(eigenvalues)
    maximum = float(np.max(real))
    return DynamicsStabilityReport(
        eigenvalues_real_max=maximum,
        eigenvalues_real_min=float(np.min(real)),
        stable=bool(maximum < -tolerance),
        tolerance=float(tolerance),
    )


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    """Software and numerical integrity checks for a simulation result."""

    finite: bool
    monotonic_time: bool
    bounded: bool
    minimum: float
    maximum: float
    boundary_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_result(result: object) -> TrajectoryDiagnostics:
    """Validate a SimulationResult-like object without depending on its class."""
    time = np.asarray(result.time, dtype=float)
    values = np.asarray(result.values, dtype=float)
    if time.ndim != 1 or values.ndim != 3 or values.shape[0] != time.size:
        raise ValueError("result has incompatible time/trajectory shapes")
    finite = bool(np.isfinite(time).all() and np.isfinite(values).all())
    monotonic = bool(time.size >= 2 and np.all(np.diff(time) > 0))
    bounded = bool(values.size == 0 or ((values >= 0).all() and (values <= 1).all()))
    boundary = float(np.mean((values <= 0) | (values >= 1))) if values.size else 0.0
    minimum = float(np.min(values)) if values.size else float("nan")
    maximum = float(np.max(values)) if values.size else float("nan")
    return TrajectoryDiagnostics(finite, monotonic, bounded, minimum, maximum, boundary)


@dataclass(frozen=True)
class ConvergenceReport:
    """Richardson-style paired comparison of coarse and refined RK4 trajectories."""

    coarse_dt: float
    refined_dt: float
    refinement_factor: int
    trajectory_rmse: float
    endpoint_rmse: float
    max_abs_error: float
    tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def timestep_convergence(
    simulator: object,
    schedule: EventSchedule | None = None,
    *,
    refinement_factor: int = 2,
    tolerance: float = 1e-4,
) -> ConvergenceReport:
    """Compare a deterministic coarse simulation with a refined RK4 simulation.

    Process noise is disabled and state clipping is disabled for this numerical
    check so the integrator is tested rather than the stochastic/clamping layers.
    The user-facing simulator configuration is not mutated.
    """
    if refinement_factor < 2:
        raise ValueError("refinement_factor must be >= 2")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    config = simulator.config
    if not isinstance(config, SimulationConfig):
        raise TypeError("simulator must expose a SimulationConfig as .config")
    coarse_config = replace(config, process_noise=0.0, clamp_states=False)
    refined_config = replace(
        config,
        dt=config.dt / refinement_factor,
        process_noise=0.0,
        clamp_states=False,
    )
    coarse = simulator.__class__(coarse_config, dynamics=simulator.dynamics).run(schedule)
    refined = simulator.__class__(refined_config, dynamics=simulator.dynamics).run(schedule)

    refined_indices = []
    for t in coarse.time:
        index = int(np.argmin(np.abs(refined.time - t)))
        if abs(float(refined.time[index]) - float(t)) > 1e-9:
            raise ValueError("coarse and refined time grids do not align")
        refined_indices.append(index)
    paired = refined.values[np.asarray(refined_indices)]
    difference = coarse.values - paired
    return ConvergenceReport(
        coarse_dt=float(config.dt),
        refined_dt=float(config.dt / refinement_factor),
        refinement_factor=int(refinement_factor),
        trajectory_rmse=float(np.sqrt(np.mean(difference**2))),
        endpoint_rmse=float(np.sqrt(np.mean(difference[-1] ** 2))),
        max_abs_error=float(np.max(np.abs(difference))),
        tolerance=float(tolerance),
        passed=bool(np.sqrt(np.mean(difference**2)) <= tolerance),
    )
