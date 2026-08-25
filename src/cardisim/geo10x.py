"""Small dependency-light readers for GEO 10X matrices.

The reader intentionally handles local GEO files only. Downloading public data
belongs in the reproducible acquisition script so scientific runs remain
explicit and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip

import numpy as np


@dataclass(frozen=True)
class TenXMatrix:
    matrix: np.ndarray
    features: tuple[str, ...]
    barcodes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.matrix.shape != (len(self.features), len(self.barcodes)):
            raise ValueError("matrix dimensions do not match features/barcodes")


def _read_lines(path: str | Path) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n\r") for line in handle]


def read_features(path: str | Path) -> tuple[str, ...]:
    rows = [x for x in _read_lines(path) if x.strip()]
    names: list[str] = []
    for row in rows:
        parts = row.split("\t")
        if len(parts) >= 2:
            names.append(parts[1])
        else:
            names.append(parts[0])
    return tuple(names)


def read_barcodes(path: str | Path) -> tuple[str, ...]:
    return tuple(x for x in _read_lines(path) if x.strip())


def read_matrix(path: str | Path, n_features: int, n_barcodes: int) -> np.ndarray:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        rows = []
        expected_shape = None
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if expected_shape is None:
                expected_shape = tuple(map(int, line.split()))
                if expected_shape != (n_features, n_barcodes):
                    raise ValueError(
                        f"MatrixMarket shape {expected_shape} disagrees with metadata "
                        f"({n_features}, {n_barcodes})"
                    )
                matrix = np.zeros(expected_shape, dtype=float)
                continue
            i, j, value = line.split()
            matrix[int(i) - 1, int(j) - 1] = float(value)
    if expected_shape is None:
        raise ValueError("MatrixMarket file is empty")
    return matrix


def read_10x(matrix_path: str | Path, features_path: str | Path, barcodes_path: str | Path) -> TenXMatrix:
    features = read_features(features_path)
    barcodes = read_barcodes(barcodes_path)
    matrix = read_matrix(matrix_path, len(features), len(barcodes))
    return TenXMatrix(matrix, features, barcodes)
