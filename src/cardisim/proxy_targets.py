"""Transparent gene-module to CardiSim phenotype proxy scoring.

Scores are deliberately presented as *proxies*. They are not clinical biomarkers
or experimentally validated latent variables. Each phenotype is represented by
a small marker module and scored from log1p-normalized expression.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .models import PHENOTYPES

DEFAULT_MODULES: Mapping[str, tuple[str, ...]] = {
    "maturity": ("MYH7", "TNNT2", "ACTN2", "RYR2", "CACNA1C"),
    "contractility": ("ACTC1", "TNNT2", "TNNI3", "MYL2", "MYH7"),
    "calcium_handling": ("RYR2", "ATP2A2", "PLN", "CACNA1C", "CASQ2"),
    "electrophysiology": ("KCNJ2", "SCN5A", "CACNA1C", "GJA1", "HCN4"),
    "metabolism": ("PPARGC1A", "CPT1B", "ACADVL", "FABP3", "PDK4"),
    "hypertrophy": ("NPPA", "NPPB", "MYH7", "ACTA1", "IGFBP3"),
    "fibrosis": ("COL1A1", "COL3A1", "DCN", "LUM", "POSTN"),
    "inflammation": ("IL1B", "TNF", "CCL2", "CXCL2", "NFKBIA"),
    "angiogenesis": ("KDR", "ESM1", "EMCN", "PECAM1", "VWF"),
    "viability": ("BCL2", "BAX", "HMOX1", "SOD2", "XIAP"),
    "oxidative_stress": ("NFE2L2", "HMOX1", "NOX4", "DUOX1", "SOD1"),
    "mitochondrial_health": ("TFAM", "PPARGC1A", "NRF1", "COX4I1", "ATP5F1A"),
}


def log1p_cpm(expression: np.ndarray) -> np.ndarray:
    """Normalize feature x cell counts by library size and log1p transform."""
    x = np.asarray(expression, dtype=float)
    if x.ndim != 2:
        raise ValueError("expression must be a 2-D feature x cell matrix")
    totals = x.sum(axis=0)
    if np.any(totals <= 0):
        raise ValueError("every cell must contain at least one count")
    return np.log1p(x / totals[None, :] * 1e4)


def module_scores(
    expression: np.ndarray,
    genes: Sequence[str],
    modules: Mapping[str, Sequence[str]] = DEFAULT_MODULES,
) -> tuple[np.ndarray, dict[str, tuple[str, ...]]]:
    """Return cell x phenotype proxy scores and the genes actually found."""
    expr = log1p_cpm(expression)
    index = {g.upper(): i for i, g in enumerate(genes)}
    scores = np.zeros((expr.shape[1], len(PHENOTYPES)), dtype=float)
    found: dict[str, tuple[str, ...]] = {}
    for j, phenotype in enumerate(PHENOTYPES):
        valid = tuple(g for g in modules[phenotype] if g.upper() in index)
        found[phenotype] = valid
        if not valid:
            continue
        rows = [index[g.upper()] for g in valid]
        scores[:, j] = expr[rows].mean(axis=0)
    # Robust within-dataset min/max scaling into the simulator's [0, 1] range.
    lo = np.nanpercentile(scores, 1, axis=0)
    hi = np.nanpercentile(scores, 99, axis=0)
    denom = np.where(hi > lo, hi - lo, 1.0)
    scores = np.clip((scores - lo[None, :]) / denom[None, :], 0.0, 1.0)
    return scores, found


def mean_target(scores: np.ndarray, weights: Sequence[float] | None = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(PHENOTYPES):
        raise ValueError("scores must have shape (cells, phenotypes)")
    return np.average(scores, axis=0, weights=None if weights is None else np.asarray(weights))
