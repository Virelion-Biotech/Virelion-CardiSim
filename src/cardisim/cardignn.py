"""Native geometry-aware graph model for the DeepCardioSim EP contract.

CardiSim does not copy the upstream model. The implementation keeps the
published five point features plus three coordinate channels (8 inputs) and a
single activation-time target, while using explicit relative geometry in graph
messages. Graph construction always uses the raw physical coordinates so the
published radius remains in physical coordinate units even when input
features are standardized.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # Optional dependency: CardiSim base installation remains NumPy-only.
    import torch
    from torch import nn
    from torch_geometric.nn import SAGEConv, radius_graph
except ImportError:  # pragma: no cover - exercised through optional-install tests
    torch = None
    nn = None
    SAGEConv = None
    radius_graph = None


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
    architecture: str = "spatial"


def _validate_config(config: CardiGNNConfig) -> None:
    if config.layers < 1:
        raise ValueError("layers must be >= 1")
    if config.in_channels != 8:
        raise ValueError("published DeepCardioSim contract uses 8 input channels")
    if config.out_channels != 1:
        raise ValueError("activation-time benchmark currently expects 1 output")
    if config.hidden_channels < 4:
        raise ValueError("hidden_channels must be >= 4")
    if config.radius <= 0:
        raise ValueError("radius must be > 0")
    if config.max_num_neighbors < 1:
        raise ValueError("max_num_neighbors must be >= 1")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if config.architecture not in {"spatial", "sage"}:
        raise ValueError("architecture must be 'spatial' or 'sage'")


if nn is not None:

    class _SpatialMessageLayer(nn.Module):
        """Message passing that exposes relative position and distance explicitly."""

        def __init__(self, hidden_channels: int, dropout: float) -> None:
            super().__init__()
            message_in = hidden_channels * 2 + 4
            self.message_mlp = nn.Sequential(
                nn.Linear(message_in, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.update_mlp = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.norm = nn.LayerNorm(hidden_channels)

        def forward(self, x, pos, edge_index):
            src, dst = edge_index
            relative = pos[src] - pos[dst]
            distance = relative.norm(dim=1, keepdim=True)
            message_input = torch.cat((x[src], x[dst], relative, distance), dim=1)
            messages = self.message_mlp(message_input)
            aggregate = torch.zeros_like(x)
            aggregate.index_add_(0, dst, messages)
            degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
            degree.index_add_(
                0, dst, torch.ones((dst.shape[0], 1), dtype=x.dtype, device=x.device)
            )
            aggregate = aggregate / degree.clamp_min(1.0)
            updated = self.update_mlp(torch.cat((x, aggregate), dim=1))
            return self.norm(x + updated)


    class CardiGNN(nn.Module):
        """Geometry-aware activation-time predictor with an optional SAGE baseline."""

        def __init__(self, config: CardiGNNConfig | None = None) -> None:
            super().__init__()
            self.config = config or CardiGNNConfig()
            _validate_config(self.config)
            self.input_projection = nn.Linear(
                self.config.in_channels, self.config.hidden_channels
            )
            self.output_head = nn.Sequential(
                nn.Linear(self.config.hidden_channels, self.config.hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_channels, self.config.out_channels),
            )
            if self.config.architecture == "sage":
                self.layers = nn.ModuleList(
                    SAGEConv(self.config.hidden_channels, self.config.hidden_channels)
                    for _ in range(self.config.layers)
                )
                self.norms = nn.ModuleList(
                    nn.LayerNorm(self.config.hidden_channels) for _ in range(self.config.layers)
                )
            else:
                self.layers = nn.ModuleList(
                    _SpatialMessageLayer(self.config.hidden_channels, self.config.dropout)
                    for _ in range(self.config.layers)
                )

        def forward(self, a, input_geom, batch=None, graph_pos=None):
            if a.ndim != 2 or a.shape[1] != 5:
                raise ValueError(f"a must have shape (N, 5), got {tuple(a.shape)}")
            if input_geom.ndim != 2 or input_geom.shape[1] != 3:
                raise ValueError(
                    "input_geom must have shape (N, 3), "
                    f"got {tuple(input_geom.shape)}"
                )
            if a.shape[0] != input_geom.shape[0]:
                raise ValueError("a and input_geom must have the same node count")
            if graph_pos is None:
                graph_pos = input_geom
            if graph_pos.ndim != 2 or graph_pos.shape != input_geom.shape:
                raise ValueError("graph_pos must have shape (N, 3) matching input_geom")
            if batch is None:
                batch = graph_pos.new_zeros(graph_pos.shape[0], dtype=torch.long)

            x = self.input_projection(torch.cat((a, input_geom), dim=1))
            neighbor_limit = self.config.max_num_neighbors
            edge_index = radius_graph(
                graph_pos,
                r=self.config.radius,
                loop=True,
                max_num_neighbors=neighbor_limit,
                batch=batch,
            )
            if self.config.architecture == "sage":
                for conv, norm in zip(self.layers, self.norms):
                    residual = x
                    x = conv(x, edge_index)
                    x = norm(x)
                    x = torch.relu(x)
                    x = x + residual
            else:
                for layer in self.layers:
                    x = layer(x, graph_pos, edge_index)
            return self.output_head(x)

else:

    class CardiGNN:  # pragma: no cover - only used when optional dependencies are absent
        """Placeholder that gives a clear installation error."""

        def __init__(self, config: CardiGNNConfig | None = None) -> None:
            del config
            raise ImportError(
                "CardiGNN requires optional dependencies: torch and torch_geometric"
            )


def canonical_model_input(a, input_geom):
    """Return the published 8-channel CardiGNN input tensor."""

    if torch is None:  # pragma: no cover - optional dependency
        raise ImportError("canonical_model_input requires torch")
    if a.ndim != 2 or a.shape[1] != 5:
        raise ValueError(f"a must have shape (N, 5), got {tuple(a.shape)}")
    if input_geom.ndim != 2 or input_geom.shape[1] != 3:
        raise ValueError(
            f"input_geom must have shape (N, 3), got {tuple(input_geom.shape)}"
        )
    if a.shape[0] != input_geom.shape[0]:
        raise ValueError("a and input_geom must have the same node count")
    return torch.cat((a, input_geom), dim=1)
