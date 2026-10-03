"""Memory-safe readers for GEO 10X MatrixMarket resources."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
from collections.abc import Iterator, Mapping, Sequence

import numpy as np

from .marker_modules import PROXY_MODULES
from .models import N_FEATURES, PHENOTYPES


@dataclass(frozen=True)
class TenXMatrix:
    matrix: np.ndarray
    features: tuple[str, ...]
    barcodes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.matrix.shape != (len(self.features), len(self.barcodes)):
            raise ValueError("matrix dimensions do not match features/barcodes")


def _open_text(path: str | Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8")


def _read_lines(path: str | Path) -> list[str]:
    with _open_text(path) as handle:
        return [line.rstrip("\n\r") for line in handle]


def read_features(path: str | Path) -> tuple[str, ...]:
    rows = [x for x in _read_lines(path) if x.strip()]
    names: list[str] = []
    for row in rows:
        parts = row.split("\t")
        names.append(parts[1] if len(parts) >= 2 else parts[0])
    return tuple(names)


def read_barcodes(path: str | Path) -> tuple[str, ...]:
    return tuple(x for x in _read_lines(path) if x.strip())


def iter_matrix_entries(path: str | Path) -> tuple[tuple[int, int], Iterator[tuple[int, int, float]]]:
    """Return sparse MatrixMarket shape and a one-pass 0-based entry iterator."""
    handle = _open_text(path)
    shape: tuple[int, int] | None = None
    for raw in handle:
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        parts = tuple(map(int, line.split()))
        if len(parts) != 3:
            handle.close()
            raise ValueError("expected MatrixMarket dimensions")
        shape = (parts[0], parts[1])
        break
    if shape is None:
        handle.close()
        raise ValueError("MatrixMarket file is empty")

    def entries() -> Iterator[tuple[int, int, float]]:
        try:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("%"):
                    continue
                i, j, value = line.split()
                yield int(i) - 1, int(j) - 1, float(value)
        finally:
            handle.close()

    return shape, entries()


def read_matrix(path: str | Path, n_features: int, n_barcodes: int) -> np.ndarray:
    """Dense reader retained for small matrices; large 10X data should use sparse_module_scores."""
    shape, entries = iter_matrix_entries(path)
    if shape != (n_features, n_barcodes):
        raise ValueError(f"MatrixMarket shape {shape} disagrees with ({n_features}, {n_barcodes})")
    matrix = np.zeros(shape, dtype=float)
    for i, j, value in entries:
        matrix[i, j] = value
    return matrix


def read_10x(matrix_path: str | Path, features_path: str | Path, barcodes_path: str | Path) -> TenXMatrix:
    features = read_features(features_path)
    barcodes = read_barcodes(barcodes_path)
    matrix = read_matrix(matrix_path, len(features), len(barcodes))
    return TenXMatrix(matrix, features, barcodes)


def sparse_module_scores(
    matrix_path: str | Path,
    features_path: str | Path,
    barcodes_path: str | Path,
    modules: Mapping[str, Sequence[str]] = PROXY_MODULES,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Score phenotype modules directly from sparse 10X counts without densifying."""
    genes = read_features(features_path)
    barcodes = read_barcodes(barcodes_path)
    index = {gene.upper(): i for i, gene in enumerate(genes)}
    found: dict[str, tuple[str, ...]] = {}
    marker_sets: dict[int, set[int]] = {}
    for phenotype_index, phenotype in enumerate(PHENOTYPES):
        present = tuple(g for g in modules[phenotype] if g.upper() in index)
        found[phenotype] = present
        marker_sets[phenotype_index] = {index[g.upper()] for g in present}

    n_cells = len(barcodes)
    totals = np.zeros(n_cells, dtype=float)
    marker_sums = np.zeros((n_cells, N_FEATURES), dtype=float)
    shape, entries = iter_matrix_entries(matrix_path)
    if shape != (len(genes), n_cells):
        raise ValueError(f"MatrixMarket shape {shape} disagrees with metadata {(len(genes), n_cells)}")

    for gene_idx, cell_idx, value in entries:
        totals[cell_idx] += value
        for phenotype_index, row_set in marker_sets.items():
            if gene_idx in row_set:
                # Per-cell library size is not known until all rows are seen, so accumulate raw counts here.
                marker_sums[cell_idx, phenotype_index] += value

    lib_factor = np.maximum(totals, 1.0)
    scores = np.zeros_like(marker_sums)
    for phenotype_index, phenotype in enumerate(PHENOTYPES):
        n_markers = max(len(found[phenotype]), 1)
        scores[:, phenotype_index] = np.log1p(marker_sums[:, phenotype_index] / lib_factor * 1e4) / n_markers

    lo = np.nanpercentile(scores, 1, axis=0)
    hi = np.nanpercentile(scores, 99, axis=0)
    scores = np.clip((scores - lo[None, :]) / np.where(hi > lo, hi - lo, 1.0)[None, :], 0.0, 1.0)
    return scores, barcodes, found
