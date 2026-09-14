"""Native GraphSAGE baseline for the published DeepCardioSim EP contract.

This is a CardiSim implementation, not a copy of the upstream DeepCardioSim
model. It preserves the published input convention: five point features plus
three spatial coordinates (8 channels total) and one activation-time target.
"""

from __future__ import annotations

from dataclasses import dataclass


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import SAGEConv, radius_graph
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "CardiGNN requires optional dependencies: torch and torch_geometric"
        ) from exc
    return torch, nn, SAGEConv, radius_graph


@dataclass(frozen=True)
class CardiGNNConfig:
    """Configuration for the native CardiGNN benchmark model."""

    in_channels: int = 8
    hidden_channels: int = 64
    layers: int = 4
    radius: float = 0.5
    max_num_neighbors: int = 128
    dropout: float = 0.0
    out_channels: int = 1


class CardiGNN:
    """GraphSAGE activation-time predictor with radius-based graph construction.

    The public object is intentionally constructed lazily so installing
    CardiSim does not require a PyTorch/PyG runtime.
    """

    def __new__(cls, config: CardiGNNConfig | None = None):
        torch, nn, SAGEConv, radius_graph = _require_torch()
        config = config or CardiGNNConfig()
        if config.layers < 1:
            raise ValueError("layers must be >= 1")
        if config.in_channels != 8:
            raise ValueError("published DeepCardioSim contract uses 8 input channels")
        if config.out_channels != 1:
            raise ValueError("activation-time benchmark currently expects 1 output")

        class _CardiGNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = config
                self.input_projection = nn.Linear(config.in_channels, config.hidden_channels)
                self.convs = nn.ModuleList(
                    SAGEConv(config.hidden_channels, config.hidden_channels)
                    for _ in range(config.layers)
                )
                self.norms = nn.ModuleList(
                    nn.BatchNorm1d(config.hidden_channels) for _ in range(config.layers)
                )
                self.output_head = nn.Sequential(
                    nn.Linear(config.hidden_channels, config.hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_channels, config.out_channels),
                )
                self._radius_graph = radius_graph

            def forward(self, a, input_geom, batch=None):
                if a.ndim != 2 or a.shape[1] != 5:
                    raise ValueError(f"a must have shape (N, 5), got {tuple(a.shape)}")
                if input_geom.ndim != 2 or input_geom.shape[1] != 3:
                    raise ValueError(
                        "input_geom must have shape (N, 3), "
                        f"got {tuple(input_geom.shape)}"
                    )
                if a.shape[0] != input_geom.shape[0]:
                    raise ValueError("a and input_geom must have the same node count")
                if batch is None:
                    batch = input_geom.new_zeros(input_geom.shape[0], dtype=torch.long)
                x = torch.cat((a, input_geom), dim=1)
                x = self.input_projection(x)
                neighbor_limit = (
                    32 if self.training else self.config.max_num_neighbors
                )
                edge_index = self._radius_graph(
                    input_geom,
                    r=self.config.radius,
                    loop=True,
                    max_num_neighbors=neighbor_limit,
                    batch=batch,
                )
                for conv, norm in zip(self.convs, self.norms):
                    residual = x
                    x = conv(x, edge_index)
                    x = norm(x)
                    x = torch.relu(x)
                    if x.shape == residual.shape:
                        x = x + residual
                return self.output_head(x)

        return _CardiGNN()


def canonical_model_input(a, input_geom):
    """Return the published 8-channel CardiGNN input tensor."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("canonical_model_input requires torch") from exc
    if a.ndim != 2 or a.shape[1] != 5:
        raise ValueError(f"a must have shape (N, 5), got {tuple(a.shape)}")
    if input_geom.ndim != 2 or input_geom.shape[1] != 3:
        raise ValueError(
            f"input_geom must have shape (N, 3), got {tuple(input_geom.shape)}"
        )
    return torch.cat((a, input_geom), dim=1)
