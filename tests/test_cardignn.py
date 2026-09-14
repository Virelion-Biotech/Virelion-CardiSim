import numpy as np
import pytest

from cardisim.deepcardiosim_data import DeepCardioSimSample


def test_cardignn_module_is_optional():
    try:
        import torch
        import torch_geometric
    except ImportError:
        pytest.skip("optional GNN dependencies are not installed")

    from cardisim.cardignn import CardiGNN, CardiGNNConfig, canonical_model_input

    a = torch.zeros((4, 5), dtype=torch.float32)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    x = canonical_model_input(a, pos)
    assert x.shape == (4, 8)

    model = CardiGNN(CardiGNNConfig(hidden_channels=8, layers=2))
    model.eval()
    with torch.no_grad():
        output = model(a, pos)
    assert output.shape == (4, 1)
    assert bool(torch.isfinite(output).all())


def test_cardignn_adapter_sample_contract():
    sample = DeepCardioSimSample(
        np.zeros((3, 3)),
        np.zeros((3, 5)),
        np.ones((3, 1)),
    )
    assert sample.n_nodes == 3
    assert sample.n_features == 5
    assert sample.n_targets == 1
