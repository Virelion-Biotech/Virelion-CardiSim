import importlib.util

import numpy as np
import pytest

from cardisim.deepcardiosim_data import DeepCardioSimSample


def test_cardignn_module_is_optional():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec(
        "torch_geometric"
    ) is None:
        pytest.skip("optional GNN dependencies are not installed")

    import torch

    from cardisim.cardignn import CardiGNN, CardiGNNConfig, canonical_model_input

    a = torch.zeros((4, 5), dtype=torch.float32)
    raw_pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    normalized_pos = (raw_pos - 0.25) / 0.5
    x = canonical_model_input(a, normalized_pos)
    assert x.shape == (4, 8)

    model = CardiGNN(CardiGNNConfig(hidden_channels=8, layers=2))
    model.eval()
    with torch.no_grad():
        output = model(a, normalized_pos, graph_pos=raw_pos)
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


def test_cardignn_case_split_is_deterministic_and_disjoint():
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec(
        "torch_geometric"
    ) is None:
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

    from scripts.train_cardignn import _split_cases

    cases = [f"case-{i}" for i in range(20)]
    split_a = _split_cases(cases, seed=7)
    split_b = _split_cases(cases, seed=7)
    train_a, val_a, test_a, idx_train, idx_val, idx_test = split_a
    train_b, val_b, test_b, _, _, _ = split_b
    assert (train_a, val_a, test_a) == (train_b, val_b, test_b)
    assert set(idx_train).isdisjoint(idx_val)
    assert set(idx_train).isdisjoint(idx_test)
    assert set(idx_val).isdisjoint(idx_test)
    assert len(train_a) + len(val_a) + len(test_a) == len(cases)
