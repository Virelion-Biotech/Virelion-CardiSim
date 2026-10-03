import numpy as np

from cardisim.geometry_reference import EPPreprocessor, UnitGaussianNormalizer


def test_normalizer_round_trip():
    values = np.asarray([[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]])
    normalizer = UnitGaussianNormalizer().fit(values)
    transformed = normalizer.transform(values)
    restored = normalizer.inverse_transform(transformed)
    assert abs(float(transformed[:, 0].mean())) < 1e-12
    assert abs(float(transformed[:, 0].std()) - 1.0) < 1e-12
    assert np.allclose(restored, values)
    assert np.all(normalizer.std[:, 1] == 1.0)


def test_ep_preprocessor_round_trip_and_grid():
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [0.0, 2.0, 1.0], [1.0, 2.0, 1.5]]
    )
    features = np.asarray([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4]])
    targets = np.asarray([[10.0], [20.0], [30.0], [40.0]])

    processor = EPPreprocessor().fit(coordinates, features, targets)
    x_norm, f_norm, y_norm = processor.transform(coordinates, features, targets)

    assert x_norm.shape == (4, 3)
    assert f_norm.shape == (4, 2)
    assert y_norm.shape == (4, 1)
    assert np.allclose(processor.inverse_target(y_norm), targets)

    grid = processor.make_query_grid(coordinates, (2, 3, 4))
    assert grid.points.shape == (24, 3)
    assert grid.shape == (2, 3, 4)
    assert np.allclose(grid.minimum, [0.0, 0.0, 0.0])
    assert np.allclose(grid.maximum, [1.0, 2.0, 1.5])


def test_ep_preprocessor_rejects_shape_mismatch():
    processor = EPPreprocessor()
    with np.testing.assert_raises_regex(ValueError, "same number of nodes"):
        processor.fit([[0.0, 0.0, 0.0]], [[1.0], [2.0]])
