from cardisim.geometry_reference import EPPreprocessor, UnitGaussianNormalizer


def test_normalizer_round_trip():
    values = [[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]]
    normalizer = UnitGaussianNormalizer().fit(values)
    transformed = normalizer.transform(values)
    restored = normalizer.inverse_transform(transformed)
    assert abs(float(transformed[:, 0].mean())) < 1e-12
    assert abs(float(transformed[:, 0].std()) - 1.0) < 1e-12
    assert (restored == values).all()
    assert (normalizer.std[:, 1] == 1.0).all()


def test_ep_preprocessor_round_trip_and_grid():
    coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [0.0, 2.0, 1.0], [1.0, 2.0, 1.5]]
    features = [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4]]
    targets = [[10.0], [20.0], [30.0], [40.0]]

    processor = EPPreprocessor().fit(coordinates, features, targets)
    x_norm, f_norm, y_norm = processor.transform(coordinates, features, targets)

    assert x_norm.shape == (4, 3)
    assert f_norm.shape == (4, 2)
    assert y_norm.shape == (4, 1)
    assert (processor.inverse_target(y_norm) == targets).all()

    grid = processor.make_query_grid(coordinates, (2, 3, 4))
    assert grid.points.shape == (24, 3)
    assert grid.shape == (2, 3, 4)
    assert (grid.minimum == [0.0, 0.0, 0.0]).all()
    assert (grid.maximum == [1.0, 2.0, 1.5]).all()


def test_ep_preprocessor_rejects_shape_mismatch():
    processor = EPPreprocessor()
    try:
        processor.fit([[0.0, 0.0, 0.0]], [[1.0], [2.0]])
    except ValueError as exc:
        assert "same number of nodes" in str(exc)
    else:
        raise AssertionError("expected a shape mismatch to raise ValueError")
