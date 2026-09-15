import numpy as np
import pytest

from cardisim.deepcardiosim_data import DeepCardioSimSample


def test_cardigino_contract_without_optional_dependency():
    try:
        import torch
    except ImportError:
        pytest.skip("optional GINO dependency is not installed")

    from cardisim.cardigino import CardiGINO, CardiGINOConfig

    config = CardiGINOConfig(
        hidden_channels=8,
        spectral_layers=1,
        grid_size=8,
        modes=(4, 4, 4),
    )
    model = CardiGINO(config)
    model.eval()
    a = torch.zeros((6, 5), dtype=torch.float32)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        output = model(a, pos)
        query_output = model(a, pos, pos[:3])
    assert model.output_head[0].in_features == 17
    assert output.shape == (6, 1)
    assert query_output.shape == (3, 1)
    assert bool(torch.isfinite(output).all())
    assert bool(torch.isfinite(query_output).all())


def test_cardigino_sample_contract():
    sample = DeepCardioSimSample(
        np.zeros((3, 3)),
        np.zeros((3, 5)),
        np.ones((3, 1)),
    )
    assert sample.n_nodes == 3
    assert sample.n_features == 5
    assert sample.n_targets == 1
