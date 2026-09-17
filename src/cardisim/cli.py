"""Command-line interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import SimulationConfig
from .phenotype_to_cdt import load_profile, uncalibrated_cdt_prior
from .presets import preset_names, population_preset
from .simulate import CardiacSimulator
from .target_derivation import write_long_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cardisim", description="Synthetic cardiac phenotype trajectory simulator")
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
    sim.add_argument("--initial-json", type=Path, help="JSON phenotype mapping used as the initial population mean")
    sim.add_argument("--cdt-output", type=Path, help="write CDT parameters derived from the final phenotype state")
    sim.add_argument("--cdt-profile", type=Path, help="JSON phenotype-to-CDT profile; defaults to an explicit uncalibrated prior")

    derive = sub.add_parser("derive-targets", help="derive latent phenotype targets from expression CSV")
    derive.add_argument("--expression", type=Path, required=True)
    derive.add_argument("--metadata", type=Path, required=True, help="JSON mapping sample -> subject_id/time")
    derive.add_argument("--dataset-id", required=True)
    derive.add_argument("--study-id", required=True)
    derive.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        initial = None
        if args.initial_json is not None:
            payload = json.loads(args.initial_json.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("--initial-json must contain a JSON object")
            initial = payload.get("phenotype_mean", payload)
        config = SimulationConfig(
            duration=args.days,
            dt=args.dt,
            n_cells=args.cells,
            seed=args.seed,
            heterogeneity=args.heterogeneity,
            process_noise=args.noise,
        )
        result = CardiacSimulator(config).run(population_preset(args.preset), initial=initial)
        if args.format == "csv":
            result.to_csv(args.output)
        else:
            result.to_json(args.output)
        if args.cdt_output is not None:
            profile = load_profile(args.cdt_profile) if args.cdt_profile is not None else uncalibrated_cdt_prior()
            payload = {
                "profile": profile.to_dict(),
                "parameters": result.cdt_parameters(profile),
                "simulation_fingerprint": result.fingerprint(),
                "warning": "Only empirically calibrated profiles should be interpreted as phenotype-dependent biological mappings.",
            }
            args.cdt_output.parent.mkdir(parents=True, exist_ok=True)
            args.cdt_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result.summary(), indent=2))
        return 0

    if args.command == "derive-targets":
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        write_long_targets(args.expression, args.output, args.dataset_id, args.study_id, metadata)
        print(f"wrote {args.output}")
        return 0

    raise RuntimeError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
