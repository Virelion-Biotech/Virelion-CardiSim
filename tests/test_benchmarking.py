import numpy as np
import pytest

from cardisim.benchmarking import aggregate_metrics, paired_differences, split_indices


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
