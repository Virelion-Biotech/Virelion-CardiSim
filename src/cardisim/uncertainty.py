"""Uncertainty utilities for CardiSim population and ensemble outputs.

These summaries quantify simulated population spread or between-run variability.
They are intentionally not reported as confidence or credible intervals unless a
separate statistical model establishes that interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .models import CardiacState, N_FEATURES, PHENOTYPES


@dataclass(frozen=True)
class PhenotypeSpread:
    """Distributional summary of one simulated population snapshot."""

    n: int
    mean: Mapping[str, float]
    std: Mapping[str, float]
    q025: Mapping[str, float]
    median: Mapping[str, float]
    q975: Mapping[str, float]
    interpretation: str = "Population spread in a synthetic simulation; not a confidence interval."

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "mean": dict(self.mean),
            "std": dict(self.std),
            "q025": dict(self.q025),
            "median": dict(self.median),
            "q975": dict(self.q975),
            "interpretation": self.interpretation,
        }


def state_spread(state: CardiacState) -> PhenotypeSpread:
    """Summarize cell-level variability for a phenotype snapshot."""
    values = np.asarray(state.values, dtype=float)
    if values.ndim != 2 or values.shape[1] != N_FEATURES or values.shape[0] == 0:
        raise ValueError("state must contain at least one population row")
    return PhenotypeSpread(
        n=int(values.shape[0]),
        mean={name: float(np.mean(values[:, i])) for i, name in enumerate(PHENOTYPES)},
        std={name: float(np.std(values[:, i], ddof=1)) if values.shape[0] > 1 else 0.0 for i, name in enumerate(PHENOTYPES)},
        q025={name: float(np.quantile(values[:, i], 0.025)) for i, name in enumerate(PHENOTYPES)},
        median={name: float(np.quantile(values[:, i], 0.5)) for i, name in enumerate(PHENOTYPES)},
        q975={name: float(np.quantile(values[:, i], 0.975)) for i, name in enumerate(PHENOTYPES)},
    )


def ensemble_final_means(results: Sequence[object]) -> PhenotypeSpread:
    """Summarize variability of final population means across independent runs.

    Each object must expose a ``final`` CardiacState property. The resulting
    standard deviations describe run-to-run variability, not posterior uncertainty.
    """
    if not results:
        raise ValueError("results cannot be empty")
    matrix = np.asarray(
        [[run.final.mean()[name] for name in PHENOTYPES] for run in results],
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[1] != N_FEATURES:
        raise ValueError("results do not expose compatible final states")
    return PhenotypeSpread(
        n=int(matrix.shape[0]),
        mean={name: float(np.mean(matrix[:, i])) for i, name in enumerate(PHENOTYPES)},
        std={name: float(np.std(matrix[:, i], ddof=1)) if matrix.shape[0] > 1 else 0.0 for i, name in enumerate(PHENOTYPES)},
        q025={name: float(np.quantile(matrix[:, i], 0.025)) for i, name in enumerate(PHENOTYPES)},
        median={name: float(np.quantile(matrix[:, i], 0.5)) for i, name in enumerate(PHENOTYPES)},
        q975={name: float(np.quantile(matrix[:, i], 0.975)) for i, name in enumerate(PHENOTYPES)},
        interpretation="Run-to-run variability across independent synthetic simulations; not a posterior interval.",
    )
