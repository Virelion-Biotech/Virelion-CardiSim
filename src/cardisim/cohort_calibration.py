"""Cohort-level calibration for destructive cardiac omics time courses."""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable

import numpy as np

from .dynamics import DynamicsParameters
from .models import N_FEATURES


@dataclass(frozen=True)
class CohortObservation:
    dataset_id: str
    study_id: str
    sample_id: str
    condition: str
    time: float
    phenotype: np.ndarray
    n_cells: int = 1

    def __post_init__(self) -> None:
        x = np.asarray(self.phenotype, dtype=float)
        if x.shape != (N_FEATURES,):
            raise ValueError("phenotype has invalid shape")
        if not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
            raise ValueError("phenotype must be finite and normalized to [0,1]")
        if self.n_cells <= 0:
            raise ValueError("n_cells must be positive")


@dataclass(frozen=True)
class CohortCalibrationResult:
    parameters: DynamicsParameters
    dataset_id: str
    study_id: str
    conditions: tuple[str, ...]
    n_samples: int
    n_transitions: int
    rmse: float
    r2: float
    regularization: float
    note: str = "Cohort-level finite differences; no repeated-cell or repeated-animal identity is assumed."

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "study_id": self.study_id,
            "conditions": list(self.conditions),
            "n_samples": self.n_samples,
            "n_transitions": self.n_transitions,
            "rmse": self.rmse,
            "r2": self.r2,
            "regularization": self.regularization,
            "note": self.note,
        }


def calibrate_cohort(observations: Iterable[CohortObservation], regularization: float = 1e-3) -> CohortCalibrationResult:
    """Fit mean cohort dynamics from independently sampled destructive timepoints."""
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    obs = list(observations)
    if not obs:
        raise ValueError("observations cannot be empty")
    dataset_ids = {o.dataset_id for o in obs}
    study_ids = {o.study_id for o in obs}
    if len(dataset_ids) != 1 or len(study_ids) != 1:
        raise ValueError("one dataset and one study are required")

    grouped: dict[tuple[str, float], list[CohortObservation]] = defaultdict(list)
    for o in obs:
        grouped[(o.condition, o.time)].append(o)

    means: dict[tuple[str, float], np.ndarray] = {}
    for key, rows in grouped.items():
        weights = np.asarray([r.n_cells for r in rows], dtype=float)
        values = np.vstack([r.phenotype for r in rows])
        means[key] = np.average(values, axis=0, weights=weights)

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for condition in sorted({o.condition for o in obs}):
        times = sorted(t for c, t in means if c == condition)
        for left_t, right_t in zip(times[:-1], times[1:]):
            dt = right_t - left_t
            if dt <= 0:
                continue
            x_rows.append(means[(condition, left_t)])
            y_rows.append((means[(condition, right_t)] - means[(condition, left_t)]) / dt)

    if not x_rows:
        raise ValueError("at least one condition with two ordered timepoints is required")

    x = np.asarray(x_rows)
    y = np.asarray(y_rows)
    design = np.hstack([np.ones((len(x), 1)), x])
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0
    theta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    pred = design @ theta
    residual = y - pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    sst = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    r2 = float(1 - np.sum(residual**2) / sst) if sst > 0 else 0.0

    params = DynamicsParameters(theta[0], theta[1:1 + N_FEATURES], np.eye(N_FEATURES), source=f"cohort:{next(iter(study_ids))}:{next(iter(dataset_ids))}")
    return CohortCalibrationResult(
        parameters=params,
        dataset_id=next(iter(dataset_ids)),
        study_id=next(iter(study_ids)),
        conditions=tuple(sorted({o.condition for o in obs})),
        n_samples=len(obs),
        n_transitions=len(x_rows),
        rmse=rmse,
        r2=r2,
        regularization=regularization,
    )
