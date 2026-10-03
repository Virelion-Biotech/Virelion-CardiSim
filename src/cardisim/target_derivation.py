"""Derive auditable latent phenotype targets from normalized expression tables."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .marker_modules import PROXY_MODULES
from .models import PHENOTYPES


def _read_expression(path: str | Path) -> tuple[list[str], list[str], np.ndarray]:
    """Read a simple CSV with columns: gene, sample1, sample2, ..."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or not rows[0] or rows[0][0].lower() not in {"gene", "gene_id"}:
        raise ValueError("expression CSV must start with a gene/gene_id column")
    samples = rows[0][1:]
    genes = []
    values = []
    for row in rows[1:]:
        if len(row) != len(samples) + 1:
            raise ValueError("inconsistent expression CSV row length")
        genes.append(row[0].upper())
        values.append([float(x) for x in row[1:]])
    matrix = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("expression matrix contains non-finite values")
    return genes, samples, matrix


def derive_targets(path: str | Path) -> tuple[list[str], dict[str, np.ndarray], dict[str, float]]:
    """Return sample IDs, normalized phenotype proxy scores, and marker coverage."""
    genes, samples, matrix = _read_expression(path)
    index = {gene: i for i, gene in enumerate(genes)}
    coverage = {
        phenotype: sum(marker in index for marker in markers) / len(markers)
        for phenotype, markers in PROXY_MODULES.items()
    }
    scores: dict[str, np.ndarray] = {}
    # Gene-wise z-scoring prevents high-abundance genes from dominating a module.
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = matrix.std(axis=1, keepdims=True)
    standardized = centered / np.where(scale > 1e-12, scale, 1.0)
    for phenotype in PHENOTYPES:
        markers = [m for m in PROXY_MODULES[phenotype] if m in index]
        if not markers:
            raise ValueError(f"no marker genes available for phenotype: {phenotype}")
        module = standardized[[index[m] for m in markers]].mean(axis=0)
        lo, hi = float(module.min()), float(module.max())
        scores[phenotype] = np.full_like(module, 0.5) if hi - lo < 1e-12 else (module - lo) / (hi - lo)
    return samples, scores, coverage


def write_long_targets(
    expression_csv: str | Path,
    output_csv: str | Path,
    dataset_id: str,
    study_id: str,
    sample_metadata: dict[str, dict[str, str]],
) -> None:
    """Convert an expression table into CardiSim calibration long format."""
    samples, scores, coverage = derive_targets(expression_csv)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset_id", "study_id", "subject_id", "time", "cell_id", "phenotype", "value", "module_coverage"])
        for sample_index, sample in enumerate(samples):
            meta = sample_metadata.get(sample)
            if meta is None:
                raise ValueError(f"missing sample metadata for {sample}")
            for phenotype in PHENOTYPES:
                writer.writerow([
                    dataset_id,
                    study_id,
                    meta["subject_id"],
                    meta["time"],
                    sample,
                    phenotype,
                    float(scores[phenotype][sample_index]),
                    float(coverage[phenotype]),
                ])
