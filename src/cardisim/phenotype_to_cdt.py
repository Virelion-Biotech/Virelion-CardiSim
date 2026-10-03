"""Calibrated bridge from CardiSim latent phenotype state to CDT parameters.

CardiSim is a phenotype-dynamics engine. This module provides an explicit,
calibration-ready interface for passing those latent states into a spatial
cardiac digital-twin model. The packaged default profile is deliberately
*uncalibrated*: it supplies only fixed parameter priors and does not claim a
biological relationship between any phenotype and an electrophysiology
parameter. Users can fit phenotype-dependent rules from paired empirical data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
import json

import numpy as np

from .models import CardiacState, N_FEATURES, PHENOTYPES

CDT_PARAMETER_NAMES = (
    "fibre_speed",
    "sheet_speed",
    "normal_speed",
    "endo_dense_speed",
    "endo_sparse_speed",
    "purkinje_speed",
    "apd_min",
    "apd_max",
)

DEFAULT_CDT_BOUNDS: dict[str, tuple[float, float]] = {
    "fibre_speed": (0.02, 0.15),
    "sheet_speed": (0.02, 0.12),
    "normal_speed": (0.02, 0.10),
    "endo_dense_speed": (0.02, 0.15),
    "endo_sparse_speed": (0.02, 0.15),
    "purkinje_speed": (0.10, 0.60),
    "apd_min": (150.0, 350.0),
    "apd_max": (180.0, 420.0),
}

DEFAULT_CDT_PRIOR: dict[str, float] = {
    "fibre_speed": 0.065,
    "sheet_speed": 0.051,
    "normal_speed": 0.048,
    "endo_dense_speed": 0.065,
    "endo_sparse_speed": 0.060,
    "purkinje_speed": 0.300,
    "apd_min": 240.0,
    "apd_max": 320.0,
}


@dataclass(frozen=True)
class CDTParameterRule:
    """One bounded affine rule centered on a neutral phenotype value of 0.5."""

    base: float
    lower: float
    upper: float
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.base):
            raise ValueError("base must be finite")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper) or self.lower >= self.upper:
            raise ValueError("CDT parameter bounds must be finite with lower < upper")
        unknown = set(self.weights) - set(PHENOTYPES)
        if unknown:
            raise ValueError(f"unknown phenotype weights: {sorted(unknown)}")
        if any(not np.isfinite(float(v)) for v in self.weights.values()):
            raise ValueError("CDT parameter weights must be finite")

    def apply(self, phenotype_mean: Mapping[str, float]) -> float:
        missing = set(PHENOTYPES) - set(phenotype_mean)
        if missing:
            raise ValueError(f"phenotype mean is missing: {sorted(missing)}")
        value = float(self.base)
        for phenotype, weight in self.weights.items():
            value += float(weight) * (float(phenotype_mean[phenotype]) - 0.5)
        return float(np.clip(value, self.lower, self.upper))


@dataclass(frozen=True)
class PhenotypeToCDTProfile:
    """Named, versioned phenotype-to-CDT mapping profile."""

    profile_id: str
    version: str
    calibration_status: str
    rules: Mapping[str, CDTParameterRule]
    source: str = "unknown"
    notes: str = ""

    def __post_init__(self) -> None:
        missing = set(CDT_PARAMETER_NAMES) - set(self.rules)
        extra = set(self.rules) - set(CDT_PARAMETER_NAMES)
        if missing or extra:
            raise ValueError(
                f"invalid CDT mapping parameters; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if not self.profile_id or not self.version:
            raise ValueError("profile_id and version must be non-empty")

    def transform(self, state: CardiacState | Mapping[str, float]) -> dict[str, float]:
        """Map a population state (or an explicit phenotype mean) into CDT parameters."""
        if isinstance(state, CardiacState):
            phenotype_mean = state.mean()
        else:
            phenotype_mean = {name: float(state[name]) for name in PHENOTYPES}
        values = {name: self.rules[name].apply(phenotype_mean) for name in CDT_PARAMETER_NAMES}
        if values["apd_max"] < values["apd_min"]:
            raise ValueError("mapped APD bounds violate apd_max >= apd_min")
        return values

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "calibration_status": self.calibration_status,
            "source": self.source,
            "notes": self.notes,
            "rules": {
                name: {
                    "base": rule.base,
                    "lower": rule.lower,
                    "upper": rule.upper,
                    "weights": dict(rule.weights),
                }
                for name, rule in self.rules.items()
            },
        }

    def save_json(self, path: str | Path) -> None:
        """Persist this mapping profile as a reviewable JSON artifact."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class MappingFitReport:
    """Fit diagnostics for a phenotype-to-CDT profile."""

    profile_id: str
    version: str
    n_samples: int
    phenotype_features: int
    parameter_metrics: Mapping[str, Mapping[str, float]]
    regularization: float
    calibration_status: str = "empirical-fit"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def profile_from_dict(payload: Mapping[str, object]) -> PhenotypeToCDTProfile:
    """Reconstruct and validate a mapping profile from JSON-compatible data."""
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, Mapping):
        raise ValueError("profile payload requires a rules mapping")
    rules: dict[str, CDTParameterRule] = {}
    for name in CDT_PARAMETER_NAMES:
        raw = raw_rules.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"profile rule for {name} is missing or invalid")
        rules[name] = CDTParameterRule(
            base=float(raw["base"]),
            lower=float(raw["lower"]),
            upper=float(raw["upper"]),
            weights={str(k): float(v) for k, v in dict(raw.get("weights") or {}).items()},
        )
    return PhenotypeToCDTProfile(
        profile_id=str(payload["profile_id"]),
        version=str(payload["version"]),
        calibration_status=str(payload.get("calibration_status", "unknown")),
        rules=rules,
        source=str(payload.get("source", "unknown")),
        notes=str(payload.get("notes", "")),
    )


def load_profile(path: str | Path) -> PhenotypeToCDTProfile:
    """Load and validate a serialized phenotype-to-CDT profile."""
    return profile_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def uncalibrated_cdt_prior(
    profile_id: str = "uncalibrated-cdt-prior",
    version: str = "1.0",
) -> PhenotypeToCDTProfile:
    """Return a fixed CDT prior with zero phenotype-dependent weights.

    This is intentionally not a disease model or biological mapping. It exists
    so CardiSim can emit a valid CDT parameter object before paired calibration.
    """
    rules = {
        name: CDTParameterRule(
            base=value,
            lower=DEFAULT_CDT_BOUNDS[name][0],
            upper=DEFAULT_CDT_BOUNDS[name][1],
            weights={},
        )
        for name, value in DEFAULT_CDT_PRIOR.items()
    }
    return PhenotypeToCDTProfile(
        profile_id=profile_id,
        version=version,
        calibration_status="uncalibrated",
        rules=rules,
        source="Virelion native CDT starting prior",
        notes="Do not interpret these fixed values as a phenotype-to-electrophysiology biological mapping.",
    )


def fit_phenotype_to_cdt(
    phenotypes: np.ndarray,
    targets: np.ndarray,
    *,
    target_names: Sequence[str] = CDT_PARAMETER_NAMES,
    regularization: float = 1e-3,
    bounds: Mapping[str, tuple[float, float]] = DEFAULT_CDT_BOUNDS,
    profile_id: str = "empirical-phenotype-to-cdt",
    version: str = "1.0",
    source: str = "paired-empirical-data",
) -> tuple[PhenotypeToCDTProfile, MappingFitReport]:
    """Fit bounded affine phenotype→CDT rules from paired observations.

    ``phenotypes`` is ``(n_samples, 12)`` in the canonical PHENOTYPES order.
    ``targets`` is ``(n_samples, 8)`` in ``target_names`` order. Fitting uses
    phenotype values centered at 0.5 so the intercept is the predicted CDT
    parameter at the neutral latent state.
    """
    x = np.asarray(phenotypes, dtype=float)
    y = np.asarray(targets, dtype=float)
    names = tuple(target_names)
    if x.ndim != 2 or x.shape[1] != N_FEATURES:
        raise ValueError(f"phenotypes must have shape (n_samples, {N_FEATURES})")
    if y.ndim != 2 or y.shape != (x.shape[0], len(names)):
        raise ValueError("targets must align with phenotypes and target_names")
    if len(names) != len(set(names)) or set(names) != set(CDT_PARAMETER_NAMES):
        raise ValueError("target_names must contain each CDT parameter exactly once")
    if x.shape[0] < 3:
        raise ValueError("at least 3 paired samples are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("phenotypes and targets must be finite")
    if np.any((x < 0) | (x > 1)):
        raise ValueError("phenotype values must be in [0, 1]")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    for name in names:
        if name not in bounds or bounds[name][0] >= bounds[name][1]:
            raise ValueError(f"missing or invalid bounds for {name}")

    design = np.column_stack((np.ones(x.shape[0]), x - 0.5))
    penalty = np.eye(design.shape[1], dtype=float) * regularization
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    theta = np.linalg.solve(lhs, rhs) if regularization > 0 else np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ theta

    rules: dict[str, CDTParameterRule] = {}
    metrics: dict[str, dict[str, float]] = {}
    for column, name in enumerate(names):
        lower, upper = bounds[name]
        rules[name] = CDTParameterRule(
            base=float(theta[0, column]),
            lower=float(lower),
            upper=float(upper),
            weights={phenotype: float(theta[i + 1, column]) for i, phenotype in enumerate(PHENOTYPES)},
        )
        residual = y[:, column] - fitted[:, column]
        sse = float(np.sum(residual**2))
        centered = y[:, column] - float(y[:, column].mean())
        sst = float(np.sum(centered**2))
        metrics[name] = {
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "r2": float(1.0 - sse / sst) if sst > 0 else 0.0,
            "predicted_min": float(fitted[:, column].min()),
            "predicted_max": float(fitted[:, column].max()),
        }

    profile = PhenotypeToCDTProfile(
        profile_id=profile_id,
        version=version,
        calibration_status="empirical-fit",
        rules=rules,
        source=source,
        notes="Fitted affine latent-state mapping; external predictive validation is required before biological interpretation.",
    )
    report = MappingFitReport(
        profile_id=profile.profile_id,
        version=profile.version,
        n_samples=int(x.shape[0]),
        phenotype_features=N_FEATURES,
        parameter_metrics=metrics,
        regularization=float(regularization),
    )
    return profile, report


def validate_phenotype_to_cdt(
    profile: PhenotypeToCDTProfile,
    phenotypes: np.ndarray,
    targets: np.ndarray,
    *,
    target_names: Sequence[str] = CDT_PARAMETER_NAMES,
) -> Mapping[str, Mapping[str, float]]:
    """Evaluate a profile on held-out paired phenotypes and CDT targets."""
    x = np.asarray(phenotypes, dtype=float)
    y = np.asarray(targets, dtype=float)
    names = tuple(target_names)
    if x.ndim != 2 or x.shape[1] != N_FEATURES or y.shape != (x.shape[0], len(names)):
        raise ValueError("held-out arrays have incompatible shapes")
    if tuple(names) != CDT_PARAMETER_NAMES:
        raise ValueError("target_names must use the canonical CDT parameter order")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("held-out arrays must be finite")
    if np.any((x < 0) | (x > 1)):
        raise ValueError("held-out phenotype values must be in [0, 1]")
    predictions = np.asarray(
        [
            [profile.rules[name].apply({phenotype: float(row[i]) for i, phenotype in enumerate(PHENOTYPES)}) for name in CDT_PARAMETER_NAMES]
            for row in x
        ],
        dtype=float,
    )
    metrics: dict[str, Mapping[str, float]] = {}
    for i, name in enumerate(names):
        residual = y[:, i] - predictions[:, i]
        sse = float(np.sum(residual**2))
        centered = y[:, i] - float(y[:, i].mean())
        sst = float(np.sum(centered**2))
        metrics[name] = {
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "r2": float(1.0 - sse / sst) if sst > 0 else 0.0,
        }
    return metrics
