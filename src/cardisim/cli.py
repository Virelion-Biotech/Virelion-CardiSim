"""Command-line interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import SimulationConfig
from .presets import preset_names, population_preset
from .simulate import CardiacSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cardisim", description="Synthetic cardiac trajectory simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run a population simulation")
    sim.add_argument("--preset", default="baseline", choices=preset_names())
    sim.add_argument("--cells", type=int, default=128)
    sim.add_argument("--days", type=float, default=28.0)
    sim.add_argument("--dt", type=float, default=0.25)
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument("--heterogeneity", type=float, default=0.05)
    sim.add_argument("--noise", type=float, default=0.003)
    sim.add_argument("--output", type=Path, required=True)
    sim.add_argument("--format", choices=("csv", "json"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        config = SimulationConfig(
            duration=args.days,
            dt=args.dt,
            n_cells=args.cells,
            seed=args.seed,
            heterogeneity=args.heterogeneity,
            process_noise=args.noise,
        )
        result = CardiacSimulator(config).run(population_preset(args.preset))
        if args.format == "csv":
            result.to_csv(args.output)
        else:
            result.to_json(args.output)
        print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
