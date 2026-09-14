import numpy as np
import pytest

from cardisim.deepcardiosim_data import DeepCardioSimSample


def make_arrays():
    geometry = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    features = np.array(
        [[1.0, 0.2, 1.0, 0.0, 0.0], [0.0, 0.2, 1.0, 0.0, 0.0],
         [0.0, 0.2, 0.0, 1.0, 0.0], [0.0, 0.2, 0.0, 0.0, 1.0]],
        dtype=float,
    )
    target = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    return geometry, features, target


def test_from_mapping_canonicalizes_target_shape():
    geometry, features, target = make_arrays()
    sample = DeepCardioSimSample.from_mapping(
        {"input_geom": geometry, "a": features, "y": target}
    )

    assert sample.input_geom.shape == (4, 3)
    assert sample.a.shape == (4, 5)
    assert sample.y.shape == (4, 1)
    assert sample.n_nodes == 4
    assert sample.n_features == 5
    assert sample.n_targets == 1


def test_from_npy_matches_upstream_column_contract(tmp_path):
    geometry, features, target = make_arrays()
    raw = np.column_stack([geometry, features, target])
    path = tmp_path / "case1_nplocs1.npy"
    np.save(path, raw)

    sample = DeepCardioSimSample.from_npy(path)

    np.testing.assert_allclose(sample.input_geom, geometry)
    np.testing.assert_allclose(sample.a, features)
    np.testing.assert_allclose(sample.y[:, 0], target)


def test_mapping_reports_missing_fields():
    with pytest.raises(KeyError, match="missing DeepCardioSim fields"):
        DeepCardioSimSample.from_mapping({"input_geom": np.zeros((1, 3))})


def test_rejects_mismatched_node_count():
    with pytest.raises(ValueError, match="same N"):
        DeepCardioSimSample(np.zeros((2, 3)), np.zeros((1, 5)), np.zeros((2, 1)))


def test_rejects_nonfinite_inputs():
    geometry = np.zeros((1, 3), dtype=float)
    features = np.zeros((1, 5), dtype=float)
    target = np.zeros((1, 1), dtype=float)
    geometry[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        DeepCardioSimSample(geometry, features, target)


def test_rejects_short_npy_representation(tmp_path):
    path = tmp_path / "invalid.npy"
    np.save(path, np.zeros((3, 8)))

    with pytest.raises(ValueError, match="N x 9\+"):
        DeepCardioSimSample.from_npy(path)
