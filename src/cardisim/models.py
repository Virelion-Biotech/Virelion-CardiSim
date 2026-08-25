"""Core state definitions and validation."""
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

    Time is expressed in arbitrary simulation days. State variables are normalized.
    """

    duration: float = 28.0
    dt: float = 0.25
    n_cells: int = 128
    seed: int = 7
    heterogeneity: float = 0.05
    process_noise: float = 0.003
    clamp_states: bool = True

    def __post_init__(self) -> None:
        if self.duration <= 0 or self.dt <= 0:
            raise ValueError("duration and dt must be positive")
        if self.n_cells <= 0:
            raise ValueError("n_cells must be positive")
        if self.duration < self.dt:
            raise ValueError("duration must be at least dt")
        if self.heterogeneity < 0 or self.process_noise < 0:
            raise ValueError("heterogeneity and process_noise must be non-negative")

    @property
    def time(self) -> np.ndarray:
        """Stable time grid including the requested final time when numerically possible."""
        n = int(np.floor(self.duration / self.dt + 1e-12))
        values = np.arange(n + 1, dtype=float) * self.dt
        if values[-1] < self.duration - 1e-10:
            values = np.append(values, self.duration)
        else:
            values[-1] = self.duration
        return values


@dataclass
class CardiacState:
    """A validated normalized cardiac phenotype vector."""

    values: np.ndarray = field(repr=False)
    cell_ids: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 2 or self.values.shape[1] != N_FEATURES:
            raise ValueError(f"values must have shape (n_cells, {N_FEATURES})")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("state contains non-finite values")
        if self.cell_ids is not None:
            self.cell_ids = np.asarray(self.cell_ids)
            if len(self.cell_ids) != len(self.values):
                raise ValueError("cell_ids length must match number of cells")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Iterable[float]]) -> "CardiacState":
        lengths = {len(np.asarray(v)) for v in mapping.values()}
        if lengths != {len(next(iter(mapping.values())))}:
            raise ValueError("all phenotype arrays must have equal length")
        missing = set(PHENOTYPES) - set(mapping)
        extra = set(mapping) - set(PHENOTYPES)
        if missing or extra:
            raise ValueError(f"invalid phenotypes; missing={sorted(missing)}, extra={sorted(extra)}")
        arr = np.column_stack([np.asarray(mapping[name], float) for name in PHENOTYPES])
        return cls(arr)

    def copy(self) -> "CardiacState":
        return CardiacState(self.values.copy(), None if self.cell_ids is None else self.cell_ids.copy())

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: self.values[:, i].copy() for i, name in enumerate(PHENOTYPES)}

    def mean(self) -> dict[str, float]:
        return {name: float(self.values[:, i].mean()) for i, name in enumerate(PHENOTYPES)}

    def clipped(self) -> "CardiacState":
        return CardiacState(np.clip(self.values, 0.0, 1.0), self.cell_ids)

    def select(self, names: Iterable[str]) -> np.ndarray:
        idx = [FEATURE_INDEX[name] for name in names]
        return self.values[:, idx]
