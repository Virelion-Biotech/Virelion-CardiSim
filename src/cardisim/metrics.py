"""Analysis helpers for generated trajectories."""
from __future__ import annotations

import numpy as np

from .models import PHENOTYPES
from .simulate import SimulationResult


def area_under_curve(result: SimulationResult, phenotype: str) -> float:
    if phenotype not in PHENOTYPES:
        raise KeyError(phenotype)
    i = PHENOTYPES.index(phenotype)
    # trapz remains available across the NumPy versions supported by pyproject.toml.
    return float(np.trapz(result.values[:, :, i].mean(axis=1), result.time))


def peak_burden(result: SimulationResult, phenotype: str, direction: str = "high") -> float:
    if phenotype not in PHENOTYPES:
        raise KeyError(phenotype)
    if direction not in {"high", "low"}:
        raise ValueError("direction must be 'high' or 'low'")
    i = PHENOTYPES.index(phenotype)
    signal = result.values[:, :, i].mean(axis=1)
    return float(signal.max() if direction == "high" else signal.min())


def recovery_fraction(result: SimulationResult, phenotype: str) -> float:
    """Return normalized return toward the initial value by the final time."""
    i = PHENOTYPES.index(phenotype)
    series = result.values[:, :, i].mean(axis=1)
    baseline = float(series[0])
    peak = float(series[1:].min())
    final = float(series[-1])
    excursion = baseline - peak
    if abs(excursion) < 1e-12:
        return 1.0
    return float(np.clip((final - peak) / excursion, 0.0, 1.0))


def group_summary(results: dict[str, SimulationResult]) -> dict[str, dict[str, float]]:
    """Compare final phenotype means across named simulation cohorts."""
    output: dict[str, dict[str, float]] = {}
    for name, result in results.items():
        output[name] = result.final.mean()
    return output
