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
        if np.asarray(self.intercept).shape != (N_FEATURES,):
            raise ValueError("intercept has invalid shape")
        if np.asarray(self.state_matrix).shape != (N_FEATURES, N_FEATURES):
            raise ValueError("state_matrix has invalid shape")
        if np.asarray(self.forcing_matrix).shape != (N_FEATURES, N_FEATURES):
            raise ValueError("forcing_matrix has invalid shape")
        if not all(np.all(np.isfinite(x)) for x in (self.intercept, self.state_matrix, self.forcing_matrix)):
            raise ValueError("dynamics parameters must be finite")

DEFAULT_PARAMETERS = DynamicsParameters(
    RELAXATION * HOMEOSTASIS - HOMEOSTASIS @ COUPLING.T,
    -np.diag(RELAXATION) + COUPLING,
    np.eye(N_FEATURES),
)


def derivative(state: np.ndarray, forcing: np.ndarray, parameters: DynamicsParameters | None = None) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    forcing = np.asarray(forcing, dtype=float)
    if state.ndim != 2 or state.shape[1] != N_FEATURES:
        raise ValueError("state has invalid shape")
    if forcing.shape != (N_FEATURES,):
        raise ValueError("forcing has invalid shape")
    params = parameters or DEFAULT_PARAMETERS
    return params.intercept[None, :] + state @ params.state_matrix.T + forcing @ params.forcing_matrix.T


def rk4_step(state: np.ndarray, t: float, dt: float, forcing_fn, parameters: DynamicsParameters | None = None) -> np.ndarray:
    k1 = derivative(state, forcing_fn(t), parameters)
    k2 = derivative(state + 0.5 * dt * k1, forcing_fn(t + 0.5 * dt), parameters)
    k3 = derivative(state + 0.5 * dt * k2, forcing_fn(t + 0.5 * dt), parameters)
    k4 = derivative(state + dt * k3, forcing_fn(t + dt), parameters)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
