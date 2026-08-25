"""Calibration of destructive time-course omics experiments.

Single-cell/single-nucleus studies generally observe different cells at each
harvest.  This module therefore fits dynamics to sample/subject-level phenotype
means instead of pretending that cell identities persist across timepoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .dynamics import DynamicsParameters
from .models import N_FEATURES, PHENOTYPES


@dataclass(frozen=True)
class Observation:
    dataset_id: str
    study_id: str
    sample_id: str
    subject_id: str
    condition: str
    time: float
    phenotype: np.ndarray
    n_cells: int = 1

    def __post_init__(self) -> None:
        x = np.asarray(self.phenotype, dtype=float)
        if x.shape != (N_FEATURES,):
            raise ValueError(f"phenotype must have shape ({N_FEATURES},)")
        if not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
            raise ValueError("phenotype must be finite and normalized to [0, 1]")
        if self.n_cells <= 0:
            raise ValueError("n_cells must be positive")


@dataclass(frozen=True)
class LongitudinalCalibrationReport:
    dataset_id: str
    study_id: str
    n_samples: int
    n_subjects: int
    n_transitions: int
    rmse: float
    r2: float
    regularization: float
    aggregation: str = "sample-level phenotype means"
    warning: str = "Destructive time-course calibration is population-level, not repeated-cell tracking."

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "study_id": self.study_id,
            "n_samples": self.n_samples,
            "n_subjects": self.n_subjects,
            "n_transitions": self.n_transitions,
            "rmse": self.rmse,
            "r2": self.r2,
            "regularization": self.regularization,
            "aggregation": self.aggregation,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class LongitudinalCalibrationResult:
    parameters: DynamicsParameters
    report: LongitudinalCalibrationReport


def _transition_rows(observations: Iterable[Observation]) -> tuple[np.ndarray, np.ndarray]:
    obs = sorted(observations, key=lambda o: (o.subject_id, o.time, o.sample_id))
    grouped: dict[str, list[Observation]] = {}
    for item in obs:
        grouped.setdefault(item.subject_id, []).append(item)

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for subject_obs in grouped.values():
        if len(subject_obs) < 2:
            continue
        for left, right in zip(subject_obs[:-1], subject_obs[1:]):
            dt = right.time - left.time
            if dt <= 0:
                continue
            x_rows.append(left.phenotype)
            y_rows.append((right.phenotype - left.phenotype) / dt)
    if not x_rows:
        raise ValueError("at least one subject with two ordered timepoints is required")
    return np.asarray(x_rows), np.asarray(y_rows)


def calibrate_longitudinal(
    observations: Iterable[Observation],
    regularization: float = 1e-3,
) -> LongitudinalCalibrationResult:
    """Fit dx/dt = b + A*x from sample/subject-level observations."""
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    obs = list(observations)
    if not obs:
        raise ValueError("observations cannot be empty")
    dataset_ids = {o.dataset_id for o in obs}
    study_ids = {o.study_id for o in obs}
    if len(dataset_ids) != 1 or len(study_ids) != 1:
        raise ValueError("calibration currently requires one dataset and one study")
    x, y = _transition_rows(obs)
    design = np.hstack([np.ones((len(x), 1)), x])
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    theta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    pred = design @ theta
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    sst = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - np.sum(resid**2) / sst) if sst > 0 else 0.0

    state_matrix = theta[1 : 1 + N_FEATURES]
    params = DynamicsParameters(
        intercept=theta[0],
        state_matrix=state_matrix,
        forcing_matrix=np.eye(N_FEATURES),
        source=f"longitudinal:{next(iter(study_ids))}:{next(iter(dataset_ids))}",
    )
    report = LongitudinalCalibrationReport(
        dataset_id=next(iter(dataset_ids)),
        study_id=next(iter(study_ids)),
        n_samples=len(obs),
        n_subjects=len({o.subject_id for o in obs}),
        n_transitions=len(x),
        rmse=rmse,
        r2=r2,
        regularization=regularization,
    )
    return LongitudinalCalibrationResult(params, report)


def phenotype_dict(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.shape != (N_FEATURES,):
        raise ValueError("values has invalid shape")
    return {name: float(values[i]) for i, name in enumerate(PHENOTYPES)}
