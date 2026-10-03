"""Geometry/feature preprocessing primitives for cardiac field benchmarks.

The implementation is intentionally NumPy-only so the preprocessing contract
can be tested without installing PyTorch, PyG, or a neural-operator stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = 1e-8


@dataclass
class UnitGaussianNormalizer:
    """Feature-wise standardization with a numerically safe inverse."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "UnitGaussianNormalizer":
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            raise ValueError("cannot fit a normalizer on an empty array")
        self.mean = np.mean(array, axis=0, keepdims=True)
        scale = np.std(array, axis=0, keepdims=True)
        self.std = np.where(scale < _EPS, 1.0, scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        array = np.asarray(values, dtype=float)
        return (array - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        array = np.asarray(values, dtype=float)
        return array * self.std + self.mean

    def _check_fitted(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("normalizer must be fitted before use")


@dataclass(frozen=True)
class QueryGrid:
    """Regular query grid generated from one irregular cardiac geometry."""

    points: np.ndarray
    shape: tuple[int, int, int]
    minimum: np.ndarray
    maximum: np.ndarray


@dataclass
class EPPreprocessor:
    """Reference-compatible preprocessing for irregular cardiac EP samples.

    Coordinates are mapped to standardized units, while each sample can also
    produce a regular query grid in its own physical bounding box. Feature and
    target normalizers are fitted explicitly rather than implicitly at inference.
    """

    coordinate_normalizer: UnitGaussianNormalizer | None = None
    feature_normalizer: UnitGaussianNormalizer | None = None
    target_normalizer: UnitGaussianNormalizer | None = None

    def fit(
        self,
        coordinates: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> "EPPreprocessor":
        coordinates = _validate_2d(coordinates, "coordinates")
        features = _validate_2d(features, "features")
        if len(coordinates) != len(features):
            raise ValueError("coordinates and features must have the same number of nodes")
        self.coordinate_normalizer = UnitGaussianNormalizer().fit(coordinates)
        self.feature_normalizer = UnitGaussianNormalizer().fit(features)
        if targets is not None:
            target_array = np.asarray(targets, dtype=float)
            self.target_normalizer = UnitGaussianNormalizer().fit(target_array)
        return self

    def transform(
        self,
        coordinates: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self.coordinate_normalizer is None or self.feature_normalizer is None:
            raise RuntimeError("preprocessor must be fitted before transform")
        coordinates = _validate_2d(coordinates, "coordinates")
        features = _validate_2d(features, "features")
        if len(coordinates) != len(features):
            raise ValueError("coordinates and features must have the same number of nodes")
        transformed_target = None
        if targets is not None:
            if self.target_normalizer is None:
                raise RuntimeError("target normalizer is not fitted")
            target_array = np.asarray(targets, dtype=float)
            transformed_target = self.target_normalizer.transform(target_array)
        return (
            self.coordinate_normalizer.transform(coordinates),
            self.feature_normalizer.transform(features),
            transformed_target,
        )

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        if self.target_normalizer is None:
            raise RuntimeError("target normalizer is not fitted")
        return self.target_normalizer.inverse_transform(values)

    @staticmethod
    def make_query_grid(
        coordinates: np.ndarray,
        resolution: tuple[int, int, int] = (28, 28, 28),
    ) -> QueryGrid:
        """Create a Cartesian query grid over a geometry's bounding box."""
        coordinates = _validate_2d(coordinates, "coordinates")
        if coordinates.shape[1] != 3:
            raise ValueError("coordinates must have exactly three spatial dimensions")
        if any(int(size) < 2 for size in resolution):
            raise ValueError("each query-grid dimension must be at least 2")
        minimum = np.min(coordinates, axis=0)
        maximum = np.max(coordinates, axis=0)
        axes = [
            np.linspace(minimum[i], maximum[i], int(resolution[i]))
            for i in range(3)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack(mesh, axis=-1).reshape(-1, 3)
        shape = tuple(map(int, resolution))
        return QueryGrid(
            points=points,
            shape=shape,
            minimum=minimum,
            maximum=maximum,
        )


def _validate_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array
