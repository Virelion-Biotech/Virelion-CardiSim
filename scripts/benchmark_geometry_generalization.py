#!/usr/bin/env python3
"""Evaluate CardiGNN/CardiGINO on a geometry-disjoint grouped split.

The runner detects repeated meshes using a canonicalized coordinate fingerprint,
then assigns complete geometry groups to train/validation/test. It also reports
how much geometry-group overlap would have occurred under a case-random split.
No assertion of geometric equivalence beyond the fingerprint definition is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from cardisim.benchmarking import (
    geometry_fingerprint,
    geometry_group_split_indices,
    split_indices,
)
from benchmark_cardigino_multiseed import (
    baseline_scores,
    case_arrays,
    fit_stats,
    load_cases,
    prepare_cases,
    repo_head,
    run_seed,
    summarize,
    torch_modules,
)


def geometry_ids(cases: list[Any], torch, decimals: int, max_points: int) -> list[str]:
    ids = []
    for index, case in enumerate(cases):
        _, pos, _ = case_arrays(case, torch)
        ids.append(
            geometry_fingerprint(
                pos.detach().cpu().numpy(),
                decimals=decimals,
                max_points=max_points,
            )
        )
        if index == 0:
            print(f"first geometry fingerprint: {ids[-1]}")
    return ids


def shared_groups(
    geometry_ids: list[str],
    train: list[int],
    validation: list[int],
    test: list[int],
) -> dict[str, int]:
    partitions = [set(train), set(validation), set(test)]
    shared = 0
    groups = Counter(geometry_ids)
    for geometry_id in groups:
        indices = [
            index
            for index, value in enumerate(geometry_ids)
            if value == geometry_id
        ]
        memberships = sum(
            any(index in partition for index in indices)
            for partition in partitions
        )
        if memberships > 1:
            shared += 1
    return {
        "groups": len(groups),
        "groups_shared_across_partitions": shared,
    }


def make_case_split(cases, geometry_ids, split_seed):
    random_train, random_val, random_test = split_indices(len(cases), split_seed)
    random_overlap = shared_groups(geometry_ids, random_train, random_val, random_test)
    train, validation, test, groups = geometry_group_split_indices(geometry_ids, split_seed)
    grouped_overlap = shared_groups(geometry_ids, train, validation, test)
    return train, validation, test, groups, random_overlap, grouped_overlap


def build_results_table(summary: dict) -> str:
    lines = [
        "| Model | Test MAE mean ± SD | Test RMSE mean ± SD | Test R² mean ± SD |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (("CardiGNN", "cardignn"), ("CardiGINO", "cardigino")):
        row = summary[key]["test_pooled"]
        lines.append(
            f"| {label} | {row['mae']['mean']:.4f} ± {row['mae']['std']:.4f} | "
            f"{row['rmse']['mean']:.4f} ± {row['rmse']['std']:.4f} | "
            f"{row['r2']['mean']:.4f} ± {row['r2']['std']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/geometry_benchmark"))
    parser.add_argument("--cases-per-shard", type=int, default=64)
    parser.add_argument("--max-cases", type=int, default=384)
    parser.add_argument("--geometry-decimals", type=int, default=4)
    parser.add_argument("--geometry-max-points", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--split-seed", type=int, default=12130875)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[12130875, 20260917, 31415927, 27182818, 8675309],
    )
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--gnn-layers", type=int, default=4)
    parser.add_argument("--gnn-radius", type=float, default=0.5)
    parser.add_argument("--gnn-max-neighbors", type=int, default=128)
    parser.add_argument("--gino-grid-size", type=int, default=16)
    parser.add_argument("--gino-modes", type=int, nargs=3, default=[8, 8, 8])
    parser.add_argument("--gino-spectral-layers", type=int, default=4)
    parser.add_argument("--gino-mlp-ratio", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.cases_per_shard < 1:
        raise ValueError("cases-per-shard must be >= 1")
    if args.max_cases < 6:
        raise ValueError("max-cases must be >= 6")
    if args.geometry_decimals < 1:
        raise ValueError("geometry-decimals must be >= 1")
    if args.geometry_max_points < 8:
        raise ValueError("geometry-max-points must be >= 8")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("model seeds must be unique")

    if args.output.exists():
        if not args.overwrite and any(args.output.iterdir()):
            raise SystemExit(
                f"Output directory {args.output} is not empty. Use --overwrite to replace it."
            )
        if args.overwrite:
            shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    torch, F = torch_modules()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_cases: list[Any] = []
    shard_counts = {}
    for shard in args.shards:
        shard_cases = load_cases(shard, args.cases_per_shard, torch)
        all_cases.extend(shard_cases)
        shard_counts[str(shard)] = len(shard_cases)

    cases = all_cases[: args.max_cases]
    if len(cases) < 6:
        raise ValueError("fewer than 6 total cases were loaded")

    ids = geometry_ids(
        cases,
        torch,
        decimals=args.geometry_decimals,
        max_points=args.geometry_max_points,
    )

    train_idx, val_idx, test_idx, groups, random_overlap, grouped_overlap = make_case_split(
        cases,
        ids,
        args.split_seed,
    )

    if grouped_overlap["groups_shared_across_partitions"] != 0:
        raise AssertionError("grouped split leaked geometry groups")

    train_raw = [cases[i] for i in train_idx]
    val_raw = [cases[i] for i in val_idx]
    test_raw = [cases[i] for i in test_idx]

    stats = fit_stats(train_raw, torch, max_nodes_per_case=4096, seed=args.split_seed)
    train_data = prepare_cases(train_raw, stats, torch)
    val_data = prepare_cases(val_raw, stats, torch)
    test_data = prepare_cases(test_raw, stats, torch)

    args.train_indices = train_idx
    args.validation_indices = val_idx
    args.test_indices = test_idx
    args.shard = Path("multiple_deepcardiosim_shards")

    print("================================================")
    print("GEOMETRY-DISJOINT CARDISIM BENCHMARK")
    print("================================================")
    print(f"device: {device}")
    print(f"repository_head: {repo_head()}")
    print(f"split_seed: {args.split_seed}")
    print(f"model_seeds: {args.seeds}")
    print(f"cases: {len(cases)}")
    print(f"source_shards: {len(args.shards)}")
    print(f"geometry_groups: {len(groups)}")
    print(f"train/validation/test: {len(train_data)}/{len(val_data)}/{len(test_data)}")
    print(
        "random_split_shared_geometry_groups: "
        f"{random_overlap['groups_shared_across_partitions']}"
    )
    print(
        "grouped_split_shared_geometry_groups: "
        f"{grouped_overlap['groups_shared_across_partitions']}"
    )

    baseline = baseline_scores(test_data, stats, torch)
    print("================================================")
    print("BASELINE TEST")
    print("================================================")
    print(json.dumps(baseline, indent=2))

    all_results = []
    for kind in ("cardignn", "cardigino"):
        for seed in args.seeds:
            print("================================================")
            print(f"RUNNING {kind.upper()} seed={seed}")
            print("================================================")
            result = run_seed(
                kind,
                seed,
                train_data,
                val_data,
                test_data,
                stats,
                args,
                torch,
                F,
                device,
            )
            all_results.append(result)
            print(
                json.dumps(
                    {
                        "model": kind,
                        "seed": seed,
                        "best_epoch": result["best_epoch"],
                        "validation": result["validation"],
                        "test": result["test"],
                        "runtime_seconds": result["runtime_seconds"],
                        "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                    },
                    indent=2,
                )
            )

    geometry_ids_hash = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    payload = {
        "benchmark": "cardisim_geometry_disjoint_cardignn_cardigino",
        "repository_head": repo_head(),
        "source_shards": [str(path) for path in args.shards],
        "source_shard_counts": shard_counts,
        "device": str(device),
        "configuration": {
            "cases_per_shard": args.cases_per_shard,
            "max_cases": args.max_cases,
            "geometry_decimals": args.geometry_decimals,
            "geometry_max_points": args.geometry_max_points,
            "epochs": args.epochs,
            "patience": args.patience,
            "split_seed": args.split_seed,
            "model_seeds": args.seeds,
            "hidden": args.hidden,
            "gnn_layers": args.gnn_layers,
            "gnn_radius": args.gnn_radius,
            "gnn_max_neighbors": args.gnn_max_neighbors,
            "gino_grid_size": args.gino_grid_size,
            "gino_modes": args.gino_modes,
            "gino_spectral_layers": args.gino_spectral_layers,
            "gino_mlp_ratio": args.gino_mlp_ratio,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "geometry_audit": {
            "total_cases": len(cases),
            "unique_geometry_groups": len(groups),
            "group_size_distribution": {
                "min": min(len(v) for v in groups.values()),
                "max": max(len(v) for v in groups.values()),
                "mean": float(np.mean([len(v) for v in groups.values()])),
                "median": float(np.median([len(v) for v in groups.values()])),
            },
            "random_case_split": random_overlap,
            "grouped_split": grouped_overlap,
            "geometry_ids_sha256": geometry_ids_hash,
            "fingerprint_definition": {
                "translation_invariant": True,
                "isotropic_scale_invariant": True,
                "row_order_invariant": True,
                "rotation_invariant": False,
                "rounded_decimals": args.geometry_decimals,
                "max_points": args.geometry_max_points,
            },
        },
        "split": {
            "train_cases": len(train_idx),
            "validation_cases": len(val_idx),
            "test_cases": len(test_idx),
            "train_indices": train_idx,
            "validation_indices": val_idx,
            "test_indices": test_idx,
        },
        "normalization": {
            "training_only": True,
            "feature_mean": stats[0].tolist(),
            "feature_std": stats[1].tolist(),
            "target_mean": stats[2].tolist(),
            "target_std": stats[3].tolist(),
        },
        "baseline": baseline,
        "results": all_results,
    }
    payload["summary"] = summarize(all_results, baseline)

    json_path = args.output / "geometry_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = (
        "# CardiSim Geometry-Disjoint Benchmark\n\n"
        "This benchmark assigns complete geometry groups to train, validation, and test partitions and compares the result against a case-random split.\n\n"
        f"- Cases: **{len(cases)}**\n"
        f"- Geometry groups: **{len(groups)}**\n"
        f"- Train/validation/test: **{len(train_idx)}/{len(val_idx)}/{len(test_idx)}**\n"
        f"- Random-split shared geometry groups: **{random_overlap['groups_shared_across_partitions']}**\n"
        f"- Grouped-split shared geometry groups: **{grouped_overlap['groups_shared_across_partitions']}**\n"
        f"- Geometry-ID SHA-256: `{geometry_ids_hash}`\n\n"
        "## Interpretation boundary\n\n"
        "A grouped split prevents a detected repeated geometry fingerprint from appearing in multiple partitions. The fingerprint is not rotation-invariant and does not prove topological equivalence, so this is a conservative mesh-reuse audit rather than a universal shape-equivalence test.\n\n"
        "## Model results\n\n"
        + build_results_table(payload["summary"])
    )
    report_path = args.output / "GEOMETRY_BENCHMARK_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    print("================================================")
    print("GEOMETRY SUMMARY")
    print("================================================")
    print(json.dumps(payload["summary"], indent=2))
    print("================================================")
    print("SAVED")
    print("================================================")
    print(json_path)
    print(report_path)
    if args.save_checkpoints:
        print(args.output / "checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
