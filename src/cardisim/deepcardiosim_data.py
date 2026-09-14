"""Model-independent adapters for the published DeepCardioSim EP sample contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DeepCardioSimSample:
    """Canonical NumPy representation of one DeepCardioSim EP sample.

    Shapes follow the upstream electrophysiology pipeline:
    ``input_geom`` is ``(N, 3)``, ``a`` is ``(N, F)``, and ``y`` is ``(N, T)``.
    The common published training representation uses ``F=5`` and ``T=1``.
    """

    input_geom: np.ndarray
    a: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        geometry = np.asarray(self.input_geom, dtype=float)
        features = np.asarray(self.a, dtype=float)
        target = np.asarray(self.y, dtype=float)

        if geometry.ndim != 2 or geometry.shape[1] != 3:
            raise ValueError(f"input_geom must have shape (N, 3), got {geometry.shape}")
        if features.ndim != 2 or features.shape[0] != geometry.shape[0]:
            raise ValueError(
                "a must have shape (N, F) with the same N as input_geom, "
                f"got {features.shape} for N={geometry.shape[0]}"
            )
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        if target.ndim != 2 or target.shape[0] != geometry.shape[0]:
            raise ValueError(
                "y must have shape (N, T) with the same N as input_geom, "
                f"got {target.shape} for N={geometry.shape[0]}"
            )
        if not np.isfinite(geometry).all():
            raise ValueError("input_geom contains non-finite values")
        if not np.isfinite(features).all():
            raise ValueError("a contains non-finite values")
        if not np.isfinite(target).all():
            raise ValueError("y contains non-finite values")

        object.__setattr__(self, "input_geom", geometry)
        object.__setattr__(self, "a", features)
        object.__setattr__(self, "y", target)

    @classmethod
    def from_mapping(cls, sample: Mapping[str, Any]) -> "DeepCardioSimSample":
        """Build a sample from the upstream ``{'input_geom', 'a', 'y'}`` mapping."""

        required = {"input_geom", "a", "y"}
        missing = required.difference(sample)
        if missing:
            raise KeyError(f"missing DeepCardioSim fields: {sorted(missing)}")
        return cls(sample["input_geom"], sample["a"], sample["y"])

    @classmethod
    def from_npy(cls, path: str | Path) -> "DeepCardioSimSample":
        """Read the upstream case ``.npy`` representation.

        The published code uses columns ``0:3`` for geometry, ``3:8`` for
        pacing/conductivity/fiber features, and ``8:9`` for activation time.
        """

        data = np.load(Path(path), allow_pickle=False)
        if data.ndim != 2 or data.shape[1] < 9:
            raise ValueError(f"expected an N x 9+ array, got {data.shape}")
        return cls(data[:, :3], data[:, 3:8], data[:, 8:9])

    @property
    def n_nodes(self) -> int:
        return int(self.input_geom.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.a.shape[1])

    @property
    def n_targets(self) -> int:
        return int(self.y.shape[1])
