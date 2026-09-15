"""Native geometry-informed neural operator for cardiac activation fields.

CardiGINO is a small, self-contained operator model inspired by the published
GINO data flow: pointwise features on an arbitrary mesh are lifted to a regular
3-D latent grid, processed by spectral operator blocks, and queried back at
arbitrary output points. It is a native CardiSim implementation, not a copy of
DeepCardioSim or neuraloperator source code.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # Optional dependency.
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None
    nn = None


@dataclass(frozen=True)
class CardiGINOConfig:
    """Configuration for the bounded native CardiGINO benchmark."""

    in_channels: int = 8
    out_channels: int = 1
    hidden_channels: int = 32
    spectral_layers: int = 4
    grid_size: int = 16
    modes: tuple[int, int, int] = (8, 8, 8)
    mlp_ratio: float = 2.0


def _validate_config(config: CardiGINOConfig) -> None:
    if config.in_channels != 8:
        raise ValueError("DeepCardioSim benchmark contract requires 8 input channels")
    if config.out_channels != 1:
        raise ValueError("cardiac activation benchmark currently expects 1 output")
    if config.hidden_channels < 4:
        raise ValueError("hidden_channels must be >= 4")
    if config.spectral_layers < 1:
        raise ValueError("spectral_layers must be >= 1")
    if config.grid_size < 4:
        raise ValueError("grid_size must be >= 4")
    if len(config.modes) != 3 or any(mode < 1 for mode in config.modes):
        raise ValueError("modes must contain three positive integers")
    if any(mode > config.grid_size // 2 for mode in config.modes):
        raise ValueError("each Fourier mode count must be <= grid_size // 2")
    if config.mlp_ratio < 1:
        raise ValueError("mlp_ratio must be >= 1")


if nn is not None:

    class _SpectralConv3d(nn.Module):
        """Low-frequency 3-D Fourier convolution with real-output symmetry."""

        def __init__(self, channels: int, modes: tuple[int, int, int]) -> None:
            super().__init__()
            self.channels = channels
            self.modes = modes
            scale = 1.0 / max(1, channels * channels)
            shape = (channels, channels, modes[0], modes[1], modes[2])
            self.weight_pos_pos = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
            self.weight_neg_pos = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
            self.weight_pos_neg = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
            self.weight_neg_neg = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

        @staticmethod
        def _multiply(x, weight):
            return torch.einsum("bixyz,ioxyz->boxyz", x, weight)

        def forward(self, x):
            size = x.shape[-3:]
            spectrum = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
            out = torch.zeros_like(spectrum)
            m1, m2, m3 = self.modes
            out[:, :, :m1, :m2, :m3] = self._multiply(
                spectrum[:, :, :m1, :m2, :m3], self.weight_pos_pos
            )
            out[:, :, -m1:, :m2, :m3] = self._multiply(
                spectrum[:, :, -m1:, :m2, :m3], self.weight_neg_pos
            )
            out[:, :, :m1, -m2:, :m3] = self._multiply(
                spectrum[:, :, :m1, -m2:, :m3], self.weight_pos_neg
            )
            out[:, :, -m1:, -m2:, :m3] = self._multiply(
                spectrum[:, :, -m1:, -m2:, :m3], self.weight_neg_neg
            )
            return torch.fft.irfftn(out, s=size, dim=(-3, -2, -1), norm="ortho")


    class _OperatorBlock(nn.Module):
        def __init__(self, channels: int, modes: tuple[int, int, int], mlp_ratio: float) -> None:
            super().__init__()
            hidden = max(channels, int(channels * mlp_ratio))
            self.spectral = _SpectralConv3d(channels, modes)
            self.local = nn.Conv3d(channels, channels, kernel_size=1)
            self.mlp = nn.Sequential(
                nn.Conv3d(channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(hidden, channels, kernel_size=1),
            )
            self.norm = nn.GroupNorm(1, channels)

        def forward(self, x):
            residual = x
            x = self.spectral(x) + self.local(x)
            x = F.gelu(x)
            x = self.mlp(x)
            return self.norm(x + residual)


    class CardiGINO(nn.Module):
        """Native geometry-informed neural operator for point-cloud fields."""

        def __init__(self, config: CardiGINOConfig | None = None) -> None:
            super().__init__()
            self.config = config or CardiGINOConfig()
            _validate_config(self.config)
            c = self.config.hidden_channels
            self.point_lift = nn.Sequential(
                nn.Linear(self.config.in_channels + 6, c),
                nn.GELU(),
                nn.Linear(c, c),
            )
            self.blocks = nn.ModuleList(
                _OperatorBlock(c, self.config.modes, self.config.mlp_ratio)
                for _ in range(self.config.spectral_layers)
            )
            self.output_head = nn.Sequential(
                nn.Linear(c + 6, c),
                nn.GELU(),
                nn.Linear(c, self.config.out_channels),
            )

        @staticmethod
        def _normalise_positions(pos):
            lo = pos.amin(dim=0)
            hi = pos.amax(dim=0)
            span = (hi - lo).clamp_min(1e-6)
            return (pos - lo) / span, lo, hi

        @staticmethod
        def _index_weights(coord, size: int):
            scaled = coord * (size - 1)
            low = torch.floor(scaled).to(torch.long).clamp(0, size - 1)
            high = (low + 1).clamp(0, size - 1)
            frac = scaled - low.to(scaled.dtype)
            return low, high, frac

        def _splat(self, values, coords):
            d = self.config.grid_size
            grid = values.new_zeros(1, values.shape[1], d, d, d)
            density = values.new_zeros(1, 1, d, d, d)
            x0, x1, fx = self._index_weights(coords[:, 0], d)
            y0, y1, fy = self._index_weights(coords[:, 1], d)
            z0, z1, fz = self._index_weights(coords[:, 2], d)
            flat = grid.view(1, values.shape[1], -1)
            flat_density = density.view(1, 1, -1)
            for ix, wx in ((x0, 1 - fx), (x1, fx)):
                for iy, wy in ((y0, 1 - fy), (y1, fy)):
                    for iz, wz in ((z0, 1 - fz), (z1, fz)):
                        weight = wx * wy * wz
                        linear = iz * d * d + iy * d + ix
                        flat.index_add_(2, linear, (values * weight[:, None]).T[None])
                        flat_density.index_add_(2, linear, weight[None, None])
            return grid / density.clamp_min(1e-6)

        def _query(self, field, coords):
            # grid_sample expects its last coordinate as x/y/z = W/H/D.
            grid = coords.view(1, -1, 1, 1, 3) * 2 - 1
            sampled = F.grid_sample(
                field,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            return sampled[0, :, :, 0, 0].T

        def forward(self, a, input_geom, output_queries=None):
            if a.ndim != 2 or a.shape[1] != 5:
                raise ValueError(f"a must have shape (N, 5), got {tuple(a.shape)}")
            if input_geom.ndim != 2 or input_geom.shape[1] != 3:
                raise ValueError(
                    f"input_geom must have shape (N, 3), got {tuple(input_geom.shape)}"
                )
            if a.shape[0] != input_geom.shape[0]:
                raise ValueError("a and input_geom must have the same node count")
            if a.shape[0] == 0:
                raise ValueError("at least one input node is required")
            if output_queries is None:
                output_queries = input_geom
            if output_queries.ndim != 2 or output_queries.shape[1] != 3:
                raise ValueError("output_queries must have shape (M, 3)")

            input_coords, lo, hi = self._normalise_positions(input_geom)
            output_coords = (output_queries - lo) / (hi - lo).clamp_min(1e-6)
            coord_embed = torch.cat(
                (
                    input_coords,
                    torch.sin(2 * torch.pi * input_coords),
                    torch.cos(2 * torch.pi * input_coords),
                ),
                dim=1,
            )
            lifted = self.point_lift(torch.cat((a, coord_embed), dim=1))
            field = self._splat(lifted, input_coords)
            for block in self.blocks:
                field = block(field)
            query_embed = torch.cat(
                (
                    output_coords,
                    torch.sin(2 * torch.pi * output_coords),
                    torch.cos(2 * torch.pi * output_coords),
                ),
                dim=1,
            )
            queried = self._query(field, output_coords)
            return self.output_head(torch.cat((queried, query_embed), dim=1))

else:

    class CardiGINO:  # pragma: no cover - optional dependency placeholder
        """Placeholder that gives a clear installation error."""

        def __init__(self, config: CardiGINOConfig | None = None) -> None:
            del config
            raise ImportError("CardiGINO requires the optional PyTorch dependency")
