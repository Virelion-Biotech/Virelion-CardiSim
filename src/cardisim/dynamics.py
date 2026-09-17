"""Phenotype dynamics for default and calibrated models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import N_FEATURES

RELAXATION = np.array([0.010, 0.020, 0.022, 0.018, 0.016, 0.012, 0.008, 0.015, 0.010, 0.030, 0.018, 0.020])
HOMEOSTASIS = np.array([0.62, 0.68, 0.65, 0.70, 0.64, 0.28, 0.08, 0.10, 0.60, 0.97, 0.12, 0.70])
COUPLING = np.zeros((N_FEATURES, N_FEATURES), dtype=float)
COUPLING[1, 2] = 0.025
COUPLING[1, 3] = 0.018
COUPLING[0, 11] = 0.015
COUPLING[4, 11] = 0.025
COUPLING[11, 10] = -0.035
COUPLING[9, 7] = -0.020
COUPLING[9, 10] = -0.025
COUPLING[6, 7] = 0.025
COUPLING[1, 6] = -0.020
COUPLING[8, 7] = 0.018


@dataclass(frozen=True)
class DynamicsParameters:
    """Linear latent dynamics parameters fitted from empirical trajectories."""

    intercept: np.ndarray
    state_matrix: np.ndarray
    forcing_matrix: np.ndarray
    source: str = "default"

    def __post_init__(self) -> None:
        intercept = np.asarray(self.intercept, dtype=float)
        state_matrix = np.asarray(self.state_matrix, dtype=float)
        forcing_matrix = np.asarray(self.forcing_matrix, dtype=float)
        if intercept.shape != (N_FEATURES,):
            raise ValueError("intercept has invalid shape")
        if state_matrix.shape != (N_FEATURES, N_FEATURES):
            raise ValueError("state_matrix has invalid shape")
        if forcing_matrix.shape != (N_FEATURES, N_FEATURES):
            raise ValueError("forcing_matrix has invalid shape")
        if not all(np.isfinite(x).all() for x in (intercept, state_matrix, forcing_matrix)):
            raise ValueError("dynamics parameters must be finite")
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "state_matrix", state_matrix)
        object.__setattr__(self, "forcing_matrix", forcing_matrix)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the homogeneous continuous-time Jacobian."""
        return np.linalg.eigvals(self.state_matrix)

    @property
    def spectral_abscissa(self) -> float:
        """Largest real eigenvalue; negative values indicate local asymptotic stability."""
        return float(np.max(np.real(self.eigenvalues)))

    def validate(self, *, require_stable: bool = False, tolerance: float = 1e-10) -> None:
        """Validate shapes/finiteness and optionally require continuous-time stability."""
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if require_stable and self.spectral_abscissa >= -tolerance:
            raise ValueError(
                "dynamics are not asymptotically stable: "
                f"spectral_abscissa={self.spectral_abscissa:.6g}"
            )


DEFAULT_PARAMETERS = DynamicsParameters(
    RELAXATION * HOMEOSTASIS - HOMEOSTASIS @ COUPLING.T,
    -np.diag(RELAXATION) + COUPLING,
    np.eye(N_FEATURES),
)


def derivative(
    state: np.ndarray,
    forcing: np.ndarray,
    parameters: DynamicsParameters | None = None,
) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    forcing = np.asarray(forcing, dtype=float)
    if state.ndim != 2 or state.shape[1] != N_FEATURES:
        raise ValueError("state has invalid shape")
    if forcing.shape != (N_FEATURES,):
        raise ValueError("forcing has invalid shape")
    if not np.isfinite(state).all() or not np.isfinite(forcing).all():
        raise ValueError("state and forcing must be finite")
    params = parameters or DEFAULT_PARAMETERS
    params.validate()
    return params.intercept[None, :] + state @ params.state_matrix.T + forcing @ params.forcing_matrix.T


def rk4_step(
    state: np.ndarray,
    t: float,
    dt: float,
    forcing_fn,
    parameters: DynamicsParameters | None = None,
) -> np.ndarray:
    """Advance one RK4 step; ``forcing_fn`` must return a 12-vector."""
    if not np.isfinite(t) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("t must be finite and dt must be finite and positive")
    k1 = derivative(state, forcing_fn(t), parameters)
    k2 = derivative(state + 0.5 * dt * k1, forcing_fn(t + 0.5 * dt), parameters)
    k3 = derivative(state + 0.5 * dt * k2, forcing_fn(t + 0.5 * dt), parameters)
    k4 = derivative(state + dt * k3, forcing_fn(t + dt), parameters)
    out = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    if not np.isfinite(out).all():
        raise FloatingPointError("RK4 step produced non-finite values")
    return out
