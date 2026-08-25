"""Phenotype dynamics for the synthetic cardiac state model."""
from __future__ import annotations

import numpy as np

from .models import FEATURE_INDEX, N_FEATURES

# Relaxation rates toward the latent homeostatic target.
RELAXATION = np.array([0.010, 0.020, 0.022, 0.018, 0.016, 0.012, 0.008, 0.015, 0.010, 0.030, 0.018, 0.020])
HOMEOSTASIS = np.array([0.62, 0.68, 0.65, 0.70, 0.64, 0.28, 0.08, 0.10, 0.60, 0.97, 0.12, 0.70])

# Hand-specified couplings encode only broad qualitative relationships.
COUPLING = np.zeros((N_FEATURES, N_FEATURES), dtype=float)
COUPLING[FEATURE_INDEX["maturity"], FEATURE_INDEX["mitochondrial_health"]] = 0.015
COUPLING[FEATURE_INDEX["contractility"], FEATURE_INDEX["calcium_handling"]] = 0.025
COUPLING[FEATURE_INDEX["contractility"], FEATURE_INDEX["electrophysiology"]] = 0.018
COUPLING[FEATURE_INDEX["metabolism"], FEATURE_INDEX["mitochondrial_health"]] = 0.025
COUPLING[FEATURE_INDEX["mitochondrial_health"], FEATURE_INDEX["oxidative_stress"]] = -0.035
COUPLING[FEATURE_INDEX["viability"], FEATURE_INDEX["inflammation"]] = -0.020
COUPLING[FEATURE_INDEX["viability"], FEATURE_INDEX["oxidative_stress"]] = -0.025
COUPLING[FEATURE_INDEX["fibrosis"], FEATURE_INDEX["inflammation"]] = 0.025
COUPLING[FEATURE_INDEX["contractility"], FEATURE_INDEX["fibrosis"]] = -0.020
COUPLING[FEATURE_INDEX["angiogenesis"], FEATURE_INDEX["inflammation"]] = 0.018


def derivative(state: np.ndarray, forcing: np.ndarray) -> np.ndarray:
    """Compute dX/dt for an `(n_cells, 12)` state matrix.

    The forcing vector is shared across cells, while heterogeneity is injected
    outside this deterministic function by the simulator.
    """
    state = np.asarray(state, dtype=float)
    if state.ndim != 2 or state.shape[1] != N_FEATURES:
        raise ValueError("state has invalid shape")
    intrinsic = (HOMEOSTASIS - state) * RELAXATION
    coupling = (state - HOMEOSTASIS) @ COUPLING.T
    # Productive vs damaging dimensions use broad bounded pressure terms.
    inflammation = state[:, FEATURE_INDEX["inflammation"]]
    oxidative = state[:, FEATURE_INDEX["oxidative_stress"]]
    protection = state[:, FEATURE_INDEX["viability"]]
    coupling[:, FEATURE_INDEX["oxidative_stress"]] += 0.012 * inflammation - 0.008 * protection
    return intrinsic + coupling + np.asarray(forcing, float)[None, :]


def rk4_step(state: np.ndarray, t: float, dt: float, forcing_fn) -> np.ndarray:
    """Advance one deterministic RK4 step."""
    k1 = derivative(state, forcing_fn(t))
    k2 = derivative(state + 0.5 * dt * k1, forcing_fn(t + 0.5 * dt))
    k3 = derivative(state + 0.5 * dt * k2, forcing_fn(t + 0.5 * dt))
    k4 = derivative(state + dt * k3, forcing_fn(t + dt))
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
