"""Model-independent adapters for the published DeepCardioSim EP sample contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PACING_NEIGHBOR_RADIUS = 0.75


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
    def from_arrays(
        cls,
        points: np.ndarray,
        point_data: Mapping[str, Any],
        *,
        pacing_neighbor_radius: float = DEFAULT_PACING_NEIGHBOR_RADIUS,
    ) -> "DeepCardioSimSample":
        """Build a sample from VTK-style points and point-data arrays.

        Required point data are pacing flag ``ploc_bool`` (optional), isotropic
        conductivity ``D_iso`` (optional), fiber direction ``ef`` (optional),
        and activation time ``activation_time`` or ``t_act``. Missing optional
        inputs are zero-filled exactly as the upstream VTK preprocessing does.
        """

        geometry = np.asarray(points, dtype=float)
        if geometry.ndim != 2 or geometry.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {geometry.shape}")
        n_nodes = geometry.shape[0]

        def column(name: str, width: int) -> np.ndarray:
            values = point_data.get(name)
            if values is None:
                return np.zeros((n_nodes, width), dtype=float)
            array = np.asarray(values, dtype=float).reshape(n_nodes, -1)
            if array.shape[1] != width:
                raise ValueError(
                    f"point_data[{name!r}] must have width {width}, got {array.shape}"
                )
            return array

        pacing = column("ploc_bool", 1)
        conductivity = column("D_iso", 1)
        fibers = column("ef", 3)
        activation_name = "activation_time" if "activation_time" in point_data else "t_act"
        if activation_name not in point_data:
            raise KeyError("point_data requires 'activation_time' or 't_act'")
        activation = column(activation_name, 1)

        pacing = _propagate_pacing_neighbors(
            geometry,
            pacing,
            radius=float(pacing_neighbor_radius),
        )
        features = np.concatenate((pacing, conductivity, fibers), axis=1)
        return cls(geometry, features, activation)

    @classmethod
    def from_vtk(
        cls,
        path: str | Path,
        *,
        pacing_neighbor_radius: float = DEFAULT_PACING_NEIGHBOR_RADIUS,
    ) -> "DeepCardioSimSample":
        """Read a published DeepCardioSim VTK case using optional ``meshio``."""

        try:
            import meshio
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise ImportError(
                "VTK loading requires optional dependency 'meshio'; install it for real-LV data"
            ) from exc

        mesh = meshio.read(Path(path))
        return cls.from_arrays(
            mesh.points,
            mesh.point_data,
            pacing_neighbor_radius=pacing_neighbor_radius,
        )

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


def _propagate_pacing_neighbors(
    points: np.ndarray,
    pacing: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    """Mark nodes within the upstream pacing-neighbor radius of each pacing node."""

    if radius < 0:
        raise ValueError("pacing_neighbor_radius must be non-negative")
    output = np.asarray(pacing, dtype=float).copy()
    pacing_nodes = np.flatnonzero(output[:, 0] == 1.0)
    if len(pacing_nodes) == 0 or radius == 0:
        return output
    radius_squared = radius * radius
    for index in pacing_nodes:
        delta = points - points[index]
        neighbors = np.einsum("ij,ij->i", delta, delta) <= radius_squared
        output[neighbors, 0] = 1.0
    return output
