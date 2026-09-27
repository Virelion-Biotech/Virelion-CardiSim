"""Pure benchmark aggregation utilities shared by reproducible runs."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
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


def geometry_fingerprint(
    points: np.ndarray,
    decimals: int = 4,
    max_points: int = 512,
) -> str:
    """Fingerprint a geometry after translation/scale normalization.

    The representation is invariant to global translation, isotropic scale,
    and point-row order. It is intentionally not rotation invariant, so it
    detects reused meshes under a shared coordinate convention rather than
    claiming universal shape or topology equivalence.
    """

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if values.shape[0] < 2:
        raise ValueError("at least 2 points are required")
    if decimals < 0:
        raise ValueError("decimals must be >= 0")
    if max_points < 2:
        raise ValueError("max_points must be >= 2")
    if not np.isfinite(values).all():
        raise ValueError("points contain non-finite values")

    raw_order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    ordered_values = values[raw_order]
    centered = ordered_values - ordered_values.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.square(centered))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("geometry has zero spatial extent")

    quantized = np.round(centered / scale, decimals=decimals)
    order = np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))
    canonical = quantized[order]

    if canonical.shape[0] > max_points:
        sample = np.linspace(
            0,
            canonical.shape[0] - 1,
            num=max_points,
            dtype=np.float64,
        ).round().astype(np.int64)
        canonical = canonical[sample]

    canonical_bytes = np.ascontiguousarray(canonical, dtype="<f4").tobytes()

    digest = hashlib.sha256()
    digest.update(f"n={values.shape[0]};d={decimals};m={max_points}".encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_bytes)
    return digest.hexdigest()


def geometry_group_split_indices(
    geometry_ids: Sequence[str],
    seed: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[list[int], list[int], list[int], dict[str, list[int]]]:
    """Split cases by geometry group so no detected group crosses partitions."""

    n_cases = len(geometry_ids)
    if n_cases < 6:
        raise ValueError("at least 6 cases are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, geometry_id in enumerate(geometry_ids):
        if not isinstance(geometry_id, str) or not geometry_id:
            raise ValueError("geometry_ids must be non-empty strings")
        groups[geometry_id].append(index)

    if len(groups) < 3:
        raise ValueError("at least 3 distinct geometry groups are required")

    n_test = max(1, round(n_cases * test_fraction))
    n_validation = max(1, round(n_cases * validation_fraction))
    n_train = n_cases - n_validation - n_test
    if n_train < 1:
        raise ValueError("split leaves no training cases")

    group_order = list(groups)
    random.Random(seed).shuffle(group_order)
    group_order.sort(key=lambda group: len(groups[group]), reverse=True)

    targets = {
        "train": n_train,
        "validation": n_validation,
        "test": n_test,
    }
    assigned: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for group in group_order:
        size = len(groups[group])
        deficits = {
            split: targets[split] - len(assigned[split])
            for split in targets
        }
        feasible = [
            split
            for split, deficit in deficits.items()
            if deficit >= size
        ]
        if feasible:
            destination = max(feasible, key=lambda split: deficits[split])
        else:
            destination = max(deficits, key=deficits.get)
        assigned[destination].extend(groups[group])

    train = sorted(assigned["train"])
    validation = sorted(assigned["validation"])
    test = sorted(assigned["test"])

    if not train or not validation or not test:
        raise ValueError("grouped split produced an empty partition")

    partition_sets = [set(train), set(validation), set(test)]
    if partition_sets[0] & partition_sets[1]:
        raise AssertionError("train/validation geometry split overlap")
    if partition_sets[0] & partition_sets[2]:
        raise AssertionError("train/test geometry split overlap")
    if partition_sets[1] & partition_sets[2]:
        raise AssertionError("validation/test geometry split overlap")
    if set.union(*partition_sets) != set(range(n_cases)):
        raise AssertionError("grouped geometry split does not cover all cases")

    for indices in groups.values():
        memberships = sum(bool(partition & set(indices)) for partition in partition_sets)
        if memberships != 1:
            raise AssertionError("a geometry group crosses split partitions")

    return train, validation, test, dict(groups)


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
