"""Core phenotype state definitions and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

PHENOTYPES = (
    "maturity",
    "contractility",
    "calcium_handling",
    "electrophysiology",
    "metabolism",
    "hypertrophy",
    "fibrosis",
    "inflammation",
    "angiogenesis",
    "viability",
    "oxidative_stress",
    "mitochondrial_health",
)
N_FEATURES = len(PHENOTYPES)
FEATURE_INDEX = {name: i for i, name in enumerate(PHENOTYPES)}


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical and population configuration.

    Time is expressed in simulation days. State variables are normalized to
    ``[0, 1]`` by the normal simulator path.
    """

    duration: float = 28.0
    dt: float = 0.25
    n_cells: int = 128
    seed: int = 7
    heterogeneity: float = 0.05
    process_noise: float = 0.003
    clamp_states: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.duration) or not np.isfinite(self.dt) or self.duration <= 0 or self.dt <= 0:
            raise ValueError("duration and dt must be finite and positive")
        if self.n_cells <= 0:
            raise ValueError("n_cells must be positive")
        if self.duration < self.dt:
            raise ValueError("duration must be at least dt")
        if self.heterogeneity < 0 or self.process_noise < 0:
            raise ValueError("heterogeneity and process_noise must be non-negative")

    @property
    def time(self) -> np.ndarray:
        """Stable time grid including the requested final time."""
        n = int(np.floor(self.duration / self.dt + 1e-12))
        values = np.arange(n + 1, dtype=float) * self.dt
        if values[-1] < self.duration - 1e-10:
            values = np.append(values, self.duration)
        else:
            values[-1] = self.duration
        return values


@dataclass
class CardiacState:
    """Validated population phenotype matrix in canonical phenotype order."""

    values: np.ndarray = field(repr=False)
    cell_ids: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2 or values.shape[1] != N_FEATURES:
            raise ValueError(f"values must have shape (n_cells, {N_FEATURES})")
        if not np.isfinite(values).all():
            raise ValueError("state contains non-finite values")
        object.__setattr__(self, "values", values)
        if self.cell_ids is not None:
            ids = np.asarray(self.cell_ids)
            if ids.ndim != 1 or len(ids) != len(values):
                raise ValueError("cell_ids must be a one-dimensional array matching number of cells")
            if len(np.unique(ids)) != len(ids):
                raise ValueError("cell_ids must be unique")
            if any(not str(item).strip() for item in ids):
                raise ValueError("cell_ids must be non-empty")
            object.__setattr__(self, "cell_ids", ids)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Iterable[float]]) -> "CardiacState":
        missing = set(PHENOTYPES) - set(mapping)
        extra = set(mapping) - set(PHENOTYPES)
        if missing or extra:
            raise ValueError(f"invalid phenotypes; missing={sorted(missing)}, extra={sorted(extra)}")
        arrays = {name: np.asarray(mapping[name], dtype=float) for name in PHENOTYPES}
        lengths = {len(values) for values in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("all phenotype arrays must have equal length")
        arr = np.column_stack([arrays[name] for name in PHENOTYPES])
        return cls(arr)

    def copy(self) -> "CardiacState":
        return CardiacState(
            self.values.copy(),
            None if self.cell_ids is None else self.cell_ids.copy(),
        )

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: self.values[:, i].copy() for i, name in enumerate(PHENOTYPES)}

    def mean(self) -> dict[str, float]:
        return {name: float(self.values[:, i].mean()) for i, name in enumerate(PHENOTYPES)}

    def mean_vector(self) -> np.ndarray:
        """Return the population mean in canonical phenotype order."""
        return self.values.mean(axis=0)

    def clipped(self) -> "CardiacState":
        return CardiacState(np.clip(self.values, 0.0, 1.0), self.cell_ids)

    def is_bounded(self) -> bool:
        """Return whether all phenotype values are inside the canonical range."""
        return bool(np.all((self.values >= 0.0) & (self.values <= 1.0)))

    def validate_bounds(self) -> None:
        """Raise when the state leaves the normalized phenotype domain."""
        if not self.is_bounded():
            raise ValueError("state values must lie in [0, 1]")

    def select(self, names: Iterable[str]) -> np.ndarray:
        names = tuple(names)
        unknown = set(names) - set(PHENOTYPES)
        if unknown:
            raise KeyError(f"unknown phenotype(s): {sorted(unknown)}")
        idx = [FEATURE_INDEX[name] for name in names]
        return self.values[:, idx]
