from __future__ import annotations

from dataclasses import asdict, dataclass
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
        times = np.asarray(self.times, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 3 or values.shape[2] != N_FEATURES:
            raise ValueError("values must have shape (time, cell, phenotype)")
        if values.shape[0] != len(times) or values.shape[1] != len(self.cell_ids):
            raise ValueError("time/cell dimensions do not match values")
        if len(self.subject_ids) != len(self.cell_ids):
            raise ValueError("subject_ids must align to cell_ids")
        if any(not str(subject).strip() for subject in self.subject_ids):
            raise ValueError("subject_ids must be non-empty")
        if any(not str(cell).strip() for cell in self.cell_ids):
            raise ValueError("cell_ids must be non-empty")
        if len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError("cell_ids must be unique")
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("empirical values must be finite and normalized to [0,1]")
        if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)):
            raise ValueError("times must be a finite one-dimensional sequence")
        if np.any(np.diff(times) <= 0):
            raise ValueError("times must be strictly increasing")
        if self.forcing is not None:
            forcing = np.asarray(self.forcing, dtype=float)
            if forcing.shape != (len(times), N_FEATURES):
                raise ValueError("forcing must have shape (time, phenotype)")
            if not np.isfinite(forcing).all():
                raise ValueError("forcing must contain only finite values")
            object.__setattr__(self, "forcing", forcing)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)


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
        payload = {
            "report": self.report.to_dict(),
            "parameters": {
                "intercept": self.parameters.intercept.tolist(),
                "state_matrix": self.parameters.state_matrix.tolist(),
                "forcing_matrix": self.parameters.forcing_matrix.tolist(),
                "source": self.parameters.source,
            },
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class SubjectHoldoutReport:
    dataset_id: str
    study_id: str
    n_subjects: int
    n_train_subjects: int
    n_test_subjects: int
    n_train_rows: int
    n_test_rows: int
    test_rmse: float
    test_r2: float
    regularization: float
    note: str = "Subjects are kept disjoint between calibration and held-out derivative validation."

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SubjectHoldoutCalibration:
    fit: CalibrationResult
    report: SubjectHoldoutReport


def _transition_design(data: EmpiricalTrajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    dt = np.diff(data.times)
    x = data.values[:-1].reshape(-1, N_FEATURES)
    y = (np.diff(data.values, axis=0) / dt[:, None, None]).reshape(-1, N_FEATURES)
    forcing = None
    if data.forcing is not None:
        forcing = np.repeat(data.forcing[:-1], data.values.shape[1], axis=0)
    return x, y, forcing


def _fit_linear(x: np.ndarray, y: np.ndarray, forcing: np.ndarray | None, regularization: float) -> tuple[np.ndarray, np.ndarray]:
    parts = [np.ones((len(x), 1)), x]
    if forcing is not None:
        parts.append(forcing)
    design = np.hstack(parts)
    penalty = np.eye(design.shape[1], dtype=float) * regularization
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    theta = np.linalg.solve(lhs, rhs) if regularization > 0 else np.linalg.lstsq(design, y, rcond=None)[0]
    return theta, design


def _report_metrics(y: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    residual = y - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    sst = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / sst) if sst > 0 else 0.0
    return rmse, r2


def load_long_csv(path: str | Path, dataset_id: str, study_id: str) -> EmpiricalTrajectory:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
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
        t = float(row["time"])
        c = row["cell_id"]
        subject = row["subject_id"]
        if c in subjects and subjects[c] != subject:
            raise ValueError(f"cell_id {c!r} is assigned to multiple subject_ids")
        subjects[c] = subject
        p = FEATURE_INDEX[row["phenotype"]]
        values[ti[t], ci[c], p] = float(row["value"])
        if forcing is not None:
            for col in force_cols:
                forcing[ti[t], FEATURE_INDEX[col[6:]]] = float(row[col])
    if np.isnan(values).any():
        raise ValueError("every cell x time x phenotype value is required")
    return EmpiricalTrajectory(
        dataset_id,
        study_id,
        np.asarray(times),
        values,
        tuple(subjects[c] for c in cells),
        tuple(cells),
        forcing,
    )


def _subset_by_subjects(data: EmpiricalTrajectory, selected: set[str]) -> EmpiricalTrajectory:
    indices = np.asarray([i for i, subject in enumerate(data.subject_ids) if subject in selected], dtype=int)
    if indices.size == 0:
        raise ValueError("subject subset is empty")
    return EmpiricalTrajectory(
        data.dataset_id,
        data.study_id,
        data.times.copy(),
        data.values[:, indices, :].copy(),
        tuple(data.subject_ids[i] for i in indices),
        tuple(data.cell_ids[i] for i in indices),
        None if data.forcing is None else data.forcing.copy(),
    )


def calibrate(data: EmpiricalTrajectory, regularization: float = 1e-3) -> CalibrationResult:
    """Fit dX/dt = b + A X + B F using ridge-regularized least squares."""
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    x, y, forcing = _transition_design(data)
    theta, design = _fit_linear(x, y, forcing, regularization)
    predicted = design @ theta
    rmse, r2 = _report_metrics(y, predicted)
    state_matrix = theta[1 : 1 + N_FEATURES]
    forcing_matrix = theta[1 + N_FEATURES :] if forcing is not None else np.eye(N_FEATURES)
    params = DynamicsParameters(
        theta[0],
        state_matrix,
        forcing_matrix,
        source=f"{data.study_id}:{data.dataset_id}",
    )
    report = CalibrationReport(
        data.dataset_id,
        data.study_id,
        len(data.cell_ids),
        len(data.times),
        len(x),
        rmse,
        r2,
        regularization,
    )
    return CalibrationResult(params, report)


def calibrate_subject_holdout(
    data: EmpiricalTrajectory,
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
    regularization: float = 1e-3,
) -> SubjectHoldoutCalibration:
    """Fit on whole subjects and assess one-step dynamics on held-out subjects."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    subjects = np.asarray(sorted(set(data.subject_ids)), dtype=object)
    if len(subjects) < 3:
        raise ValueError("at least 3 distinct subjects are required for subject holdout")
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    n_test = min(n_test, len(shuffled) - 2)
    test_subjects = set(str(x) for x in shuffled[:n_test])
    train_subjects = set(str(x) for x in shuffled[n_test:])
    train_data = _subset_by_subjects(data, train_subjects)
    test_data = _subset_by_subjects(data, test_subjects)
    fit = calibrate(train_data, regularization=regularization)

    x, y, forcing = _transition_design(test_data)
    predicted = fit.parameters.intercept[None, :] + x @ fit.parameters.state_matrix.T
    if forcing is not None:
        predicted = predicted + forcing @ fit.parameters.forcing_matrix.T
    test_rmse, test_r2 = _report_metrics(y, predicted)
    report = SubjectHoldoutReport(
        dataset_id=data.dataset_id,
        study_id=data.study_id,
        n_subjects=len(subjects),
        n_train_subjects=len(train_subjects),
        n_test_subjects=len(test_subjects),
        n_train_rows=int(len(_transition_design(train_data)[0])),
        n_test_rows=int(len(x)),
        test_rmse=test_rmse,
        test_r2=test_r2,
        regularization=regularization,
    )
    return SubjectHoldoutCalibration(fit, report)


def calibration_from_csv(path: str | Path, dataset_id: str, study_id: str, regularization: float = 1e-3) -> CalibrationResult:
    return calibrate(load_long_csv(path, dataset_id, study_id), regularization)
