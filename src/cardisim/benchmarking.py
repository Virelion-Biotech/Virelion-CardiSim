"""Pure benchmark aggregation utilities shared by reproducible runs."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np


def split_indices(
    n_cases: int,
    seed: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """Deterministically split case indices into train/validation/test sets."""

    if n_cases < 6:
        raise ValueError("at least 6 cases are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")

    indices = list(range(n_cases))
    random.Random(seed).shuffle(indices)

    n_test = max(1, round(n_cases * test_fraction))
    n_validation = max(1, round(n_cases * validation_fraction))
    n_train = n_cases - n_validation - n_test

    if n_train < 1:
        raise ValueError("split leaves no training cases")

    return (
        indices[:n_train],
        indices[n_train : n_train + n_validation],
        indices[n_train + n_validation :],
    )


def aggregate_metrics(
    records: Sequence[Mapping[str, float]],
    metrics: Sequence[str] = ("mae", "rmse", "r2"),
) -> dict[str, dict[str, float | int]]:
    """Return count/mean/std/median/min/max for each metric across seeds."""

    if not records:
        raise ValueError("cannot aggregate an empty record sequence")

    result: dict[str, dict[str, float | int]] = {}
    for metric in metrics:
        values = np.asarray(
            [float(record[metric]) for record in records],
            dtype=np.float64,
        )
        result[metric] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def paired_differences(
    left_records: Sequence[Mapping[str, float]],
    right_records: Sequence[Mapping[str, float]],
    metrics: Sequence[str] = ("mae", "rmse", "r2"),
) -> dict[str, dict[str, float | int]]:
    """Summarize paired right-minus-left differences by seed."""

    if len(left_records) != len(right_records):
        raise ValueError("paired record sequences must have equal length")
    if not left_records:
        raise ValueError("cannot compare empty record sequences")

    result: dict[str, dict[str, float | int]] = {}
    for metric in metrics:
        delta = np.asarray(
            [
                float(right[metric]) - float(left[metric])
                for left, right in zip(left_records, right_records)
            ],
            dtype=np.float64,
        )
        result[metric] = {
            "n": int(delta.size),
            "mean": float(delta.mean()),
            "std": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
            "median": float(np.median(delta)),
            "min": float(delta.min()),
            "max": float(delta.max()),
        }
    return result
