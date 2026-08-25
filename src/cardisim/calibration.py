from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json

import numpy as np

from .dynamics import DynamicsParameters
from .models import FEATURE_INDEX, N_FEATURES, PHENOTYPES

REQUIRED_COLUMNS = {"dataset_id", "study_id", "subject_id", "time", "cell_id", "phenotype", "value"}

@dataclass(frozen=True)
class EmpiricalTrajectory:
    dataset_id: str
    study_id: str
    times: np.ndarray
    values: np.ndarray
    subject_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    forcing: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[2] != N_FEATURES:
            raise ValueError("values must have shape (time, cell, phenotype)")
        if self.values.shape[0] != len(self.times) or self.values.shape[1] != len(self.cell_ids):
            raise ValueError("time/cell dimensions do not match values")
        if len(self.subject_ids) != len(self.cell_ids):
            raise ValueError("subject_ids must align to cell_ids")
        if not np.all(np.isfinite(self.values)) or np.any((self.values < 0) | (self.values > 1)):
            raise ValueError("empirical values must be finite and normalized to [0,1]")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing")
        if self.forcing is not None and self.forcing.shape != (len(self.times), N_FEATURES):
            raise ValueError("forcing must have shape (time, phenotype)")

@dataclass(frozen=True)
class CalibrationReport:
    dataset_id: str
    study_id: str
    n_cells: int
    n_timepoints: int
    n_training_rows: int
    rmse: float
    r2: float
    regularization: float
    parameter_source: str = "empirical-fit"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True)
class CalibrationResult:
    parameters: DynamicsParameters
    report: CalibrationReport

    def save_json(self, path: str | Path) -> None:
        payload = {"report": self.report.to_dict(), "parameters": {
            "intercept": self.parameters.intercept.tolist(),
            "state_matrix": self.parameters.state_matrix.tolist(),
            "forcing_matrix": self.parameters.forcing_matrix.tolist(),
            "source": self.parameters.source,
        }}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

def load_long_csv(path: str | Path, dataset_id: str, study_id: str) -> EmpiricalTrajectory:
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("empirical CSV is empty")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    unknown = {r["phenotype"] for r in rows} - set(PHENOTYPES)
    if unknown:
        raise ValueError(f"unknown phenotypes: {sorted(unknown)}")
    times = sorted({float(r["time"]) for r in rows})
    cells = sorted({r["cell_id"] for r in rows})
    ti = {t: i for i, t in enumerate(times)}
    ci = {c: i for i, c in enumerate(cells)}
    values = np.full((len(times), len(cells), N_FEATURES), np.nan)
    subjects: dict[str, str] = {}
    force_cols = [f"force_{p}" for p in PHENOTYPES if f"force_{p}" in rows[0]]
    forcing = np.zeros((len(times), N_FEATURES), dtype=float) if force_cols else None
    for row in rows:
        if row["dataset_id"] != dataset_id or row["study_id"] != study_id:
            raise ValueError("CSV contains rows from a different study or dataset")
        t, c, p = float(row["time"]), row["cell_id"], FEATURE_INDEX[row["phenotype"]]
        values[ti[t], ci[c], p] = float(row["value"])
        subjects[c] = row["subject_id"]
        if forcing is not None:
            for col in force_cols:
                forcing[ti[t], FEATURE_INDEX[col[6:]]] = float(row[col])
    if np.isnan(values).any():
        raise ValueError("every cell x time x phenotype value is required")
    return EmpiricalTrajectory(dataset_id, study_id, np.asarray(times), values, tuple(subjects[c] for c in cells), tuple(cells), forcing)

def calibrate(data: EmpiricalTrajectory, regularization: float = 1e-3) -> CalibrationResult:
    """Fit dX/dt = b + A X + B F using ridge-regularized least squares."""
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    dt = np.diff(data.times)
    y = (np.diff(data.values, axis=0) / dt[:, None, None]).reshape(-1, N_FEATURES)
    x = data.values[:-1].reshape(-1, N_FEATURES)
    parts = [np.ones((len(x), 1)), x]
    if data.forcing is not None:
        parts.append(np.repeat(data.forcing[:-1], data.values.shape[1], axis=0))
    design = np.hstack(parts)
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    theta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    predicted = design @ theta
    residual = y - predicted
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sst = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    r2 = float(1 - np.sum(residual ** 2) / sst) if sst > 0 else 0.0
    state_matrix = theta[1:1 + N_FEATURES]
    forcing_matrix = theta[1 + N_FEATURES:] if data.forcing is not None else np.eye(N_FEATURES)
    params = DynamicsParameters(theta[0], state_matrix, forcing_matrix, source=f"{data.study_id}:{data.dataset_id}")
    report = CalibrationReport(data.dataset_id, data.study_id, len(data.cell_ids), len(data.times), len(x), rmse, r2, regularization)
    return CalibrationResult(params, report)

def calibration_from_csv(path: str | Path, dataset_id: str, study_id: str, regularization: float = 1e-3) -> CalibrationResult:
    return calibrate(load_long_csv(path, dataset_id, study_id), regularization)
