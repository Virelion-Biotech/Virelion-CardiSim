import numpy as np
import pytest

from cardisim.benchmarking import (
    aggregate_metrics,
    geometry_fingerprint,
    geometry_group_split_indices,
    paired_differences,
    split_indices,
)


def test_split_indices_is_deterministic_and_disjoint():
    first = split_indices(128, seed=12130875)
    second = split_indices(128, seed=12130875)

    assert first == second

    train, validation, test = first
    assert len(train) == 90
    assert len(validation) == 19
    assert len(test) == 19

    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert len(set(train) | set(validation) | set(test)) == 128


def test_split_indices_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        split_indices(128, seed=1, validation_fraction=0.0)

    with pytest.raises(ValueError):
        split_indices(128, seed=1, validation_fraction=0.6, test_fraction=0.5)

    with pytest.raises(ValueError):
        split_indices(5, seed=1)


def test_aggregate_metrics_reports_sample_statistics():
    records = [
        {"mae": 1.0, "rmse": 2.0, "r2": 0.5},
        {"mae": 3.0, "rmse": 4.0, "r2": 0.7},
        {"mae": 5.0, "rmse": 6.0, "r2": 0.9},
    ]

    result = aggregate_metrics(records)

    assert result["mae"]["n"] == 3
    assert result["mae"]["mean"] == 3.0
    assert np.isclose(result["mae"]["std"], 2.0)
    assert result["mae"]["median"] == 3.0
    assert result["mae"]["min"] == 1.0
    assert result["mae"]["max"] == 5.0


def test_paired_differences_are_right_minus_left():
    left = [
        {"mae": 10.0, "rmse": 12.0, "r2": 0.4},
        {"mae": 8.0, "rmse": 10.0, "r2": 0.5},
    ]
    right = [
        {"mae": 6.0, "rmse": 9.0, "r2": 0.8},
        {"mae": 5.0, "rmse": 8.0, "r2": 0.7},
    ]

    result = paired_differences(left, right)

    assert result["mae"]["mean"] == -3.5
    assert result["rmse"]["mean"] == -2.5
    assert result["r2"]["mean"] == 0.3



def test_geometry_fingerprint_is_invariant_to_translation_scale_and_row_order():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.5, 0.25],
        ],
        dtype=np.float64,
    )
    transformed = points * 3.7 + np.array([11.0, -4.0, 2.0])
    transformed = transformed[[4, 2, 0, 5, 1, 3]]

    assert geometry_fingerprint(points) == geometry_fingerprint(transformed)


def test_geometry_fingerprint_large_mesh_is_row_order_invariant():
    axis = np.linspace(-2.0, 2.0, 11)
    points = np.array(
        [(x, y, z) for x in axis for y in axis for z in axis],
        dtype=np.float64,
    )
    reordered = points[np.arange(points.shape[0])[::-1]]

    first = geometry_fingerprint(points, max_points=128)
    second = geometry_fingerprint(reordered, max_points=128)
    assert first == second


def test_geometry_fingerprint_detects_distinct_geometry():
    first = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    second = first.copy()
    second[3, 2] = 2.0

    assert geometry_fingerprint(first) != geometry_fingerprint(second)


def test_geometry_group_split_never_separates_a_group():
    geometry_ids = [
        "a", "a", "a",
        "b", "b",
        "c", "c", "c",
        "d", "d",
        "e", "e",
        "f", "f", "f",
        "g", "g",
        "h", "h", "h",
    ]

    train, validation, test, groups = geometry_group_split_indices(
        geometry_ids,
        seed=12130875,
    )

    partitions = [set(train), set(validation), set(test)]
    for indices in groups.values():
        memberships = sum(bool(partition & set(indices)) for partition in partitions)
        assert memberships == 1

    assert set(train) | set(validation) | set(test) == set(range(len(geometry_ids)))
