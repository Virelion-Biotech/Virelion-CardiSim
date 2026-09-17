#!/usr/bin/env python3
"""Run a multi-seed, longer-training CardiGNN/CardiGINO comparison.

The benchmark fixes the case split and training-only normalization across all
model seeds so seed-to-seed variation measures training/model stochasticity
rather than changing the held-out population. It reports both pooled-node
metrics and case-macro metrics, plus per-seed convergence, runtime, memory,
and paired CardiGINO-minus-CardiGNN differences.

This is a bounded engineering benchmark, not an exact reproduction of the
published DeepCardioSim training protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from cardisim.benchmarking import aggregate_metrics, paired_differences, split_indices
from cardisim.cardigino import CardiGINO, CardiGINOConfig
from cardisim.cardignn import CardiGNN, CardiGNNConfig


def torch_modules():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Install PyTorch first: pip install 'virelion-cardisim[gino]'."
        ) from exc

    return torch, F


def seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def repo_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def load_cases(path: Path, max_samples: int, torch) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, list):
        raise ValueError(
            f"expected a list of cases in {path}, got {type(payload)!r}"
        )

    cases = payload[:max_samples]
    if len(cases) < 6:
        raise ValueError("at least 6 cases are required")

    return cases


def case_arrays(case: Any, torch):
    if isinstance(case, dict):
        a = case["a"]
        pos = case["input_geom"]
        y = case["y"]
    else:
        a = case.a
        pos = case.input_geom
        y = case.y

    a = torch.as_tensor(a, dtype=torch.float32)
    pos = torch.as_tensor(pos, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32).reshape(pos.shape[0], -1)

    if a.shape != (pos.shape[0], 5):
        raise ValueError(f"expected a=(N,5), got {tuple(a.shape)}")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"expected input_geom=(N,3), got {tuple(pos.shape)}")
    if y.shape != (pos.shape[0], 1):
        raise ValueError(f"expected y=(N,1), got {tuple(y.shape)}")
    if not bool(
        torch.isfinite(a).all()
        and torch.isfinite(pos).all()
        and torch.isfinite(y).all()
    ):
        raise ValueError("case contains non-finite values")

    return a, pos, y


def fit_stats(train_cases, torch, max_nodes_per_case: int, seed: int):
    rng = random.Random(seed)
    features = []
    targets = []

    for case in train_cases:
        a, pos, y = case_arrays(case, torch)
        take = min(pos.shape[0], max_nodes_per_case)
        indices = list(range(pos.shape[0]))
        rng.shuffle(indices)
        idx = torch.as_tensor(indices[:take], dtype=torch.long)
        features.append(torch.cat((a[idx], pos[idx]), dim=1))
        targets.append(y[idx])

    feature_values = torch.cat(features, dim=0)
    target_values = torch.cat(targets, dim=0)

    feature_mean = feature_values.mean(dim=0)
    feature_std = feature_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    target_mean = target_values.mean(dim=0)
    target_std = target_values.std(dim=0, unbiased=False).clamp_min(1e-6)

    return feature_mean, feature_std, target_mean, target_std


def prepare_cases(cases, stats, torch):
    feature_mean, feature_std, target_mean, target_std = stats
    prepared = []

    for case in cases:
        a, raw_pos, y = case_arrays(case, torch)
        normalized_a = (a - feature_mean[:5]) / feature_std[:5]
        normalized_pos = (raw_pos - feature_mean[5:]) / feature_std[5:]
        normalized_y = (y - target_mean) / target_std
        prepared.append((normalized_a, normalized_pos, raw_pos, normalized_y))

    return prepared


def inverse_target(values, stats):
    return values * stats[3] + stats[2]


def metrics(pred, target):
    error = pred - target
    mse = float(error.square().mean().item())
    mae = float(error.abs().mean().item())
    rmse = mse**0.5
    ss_res = float(error.square().sum().item())
    centered = target - target.mean()
    ss_tot = float(centered.square().sum().item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_model(model, cases, stats, device, torch, kind: str):
    model.eval()
    pooled_predictions = []
    pooled_targets = []
    case_scores = []

    with torch.no_grad():
        for a, normalized_pos, raw_pos, normalized_y in cases:
            a = a.to(device)
            normalized_pos = normalized_pos.to(device)
            raw_pos = raw_pos.to(device)

            if kind == "cardignn":
                prediction = model(
                    a,
                    normalized_pos,
                    graph_pos=raw_pos,
                )
            else:
                prediction = model(a, normalized_pos)

            prediction = inverse_target(prediction.cpu(), stats)
            target = inverse_target(normalized_y.cpu(), stats)

            pooled_predictions.append(prediction)
            pooled_targets.append(target)
            case_scores.append(metrics(prediction, target))

    pooled = metrics(
        torch.cat(pooled_predictions, dim=0),
        torch.cat(pooled_targets, dim=0),
    )

    macro = {
        metric: float(
            np.nanmean([score[metric] for score in case_scores])
        )
        for metric in ("mae", "rmse", "r2")
    }

    return {
        "pooled": pooled,
        "case_macro": macro,
        "n_cases": len(cases),
        "n_nodes": int(sum(score.shape[0] for score in pooled_targets)),
    }


def baseline_scores(cases, stats, torch):
    predictions = []
    targets = []
    case_scores = []

    for _, _, _, normalized_y in cases:
        target = inverse_target(normalized_y.cpu(), stats)
        prediction = stats[2].expand_as(target)
        predictions.append(prediction)
        targets.append(target)
        case_scores.append(metrics(prediction, target))

    pooled = metrics(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
    )
    macro = {
        metric: float(
            np.nanmean([score[metric] for score in case_scores])
        )
        for metric in ("mae", "rmse", "r2")
    }

    return {
        "pooled": pooled,
        "case_macro": macro,
        "n_cases": len(cases),
        "n_nodes": int(sum(target.shape[0] for target in targets)),
    }


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_one(
    model,
    train_cases,
    val_cases,
    stats,
    device,
    torch,
    F,
    kind: str,
    seed: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = None
    best_val_rmse = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_cases)))
        random.Random(seed + epoch).shuffle(order)
        loss_sum = 0.0

        for index in order:
            a, normalized_pos, raw_pos, target = train_cases[index]
            a = a.to(device)
            normalized_pos = normalized_pos.to(device)
            raw_pos = raw_pos.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)

            if kind == "cardignn":
                prediction = model(
                    a,
                    normalized_pos,
                    graph_pos=raw_pos,
                )
            else:
                prediction = model(a, normalized_pos)

            loss = F.mse_loss(prediction, target)

            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"{kind} non-finite loss at epoch {epoch}, seed {seed}"
                )

            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())

        validation = evaluate_model(
            model,
            val_cases,
            stats,
            device,
            torch,
            kind,
        )

        record = {
            "epoch": epoch,
            "train_mse_normalized": loss_sum / max(len(train_cases), 1),
            "val_mae": validation["pooled"]["mae"],
            "val_rmse": validation["pooled"]["rmse"],
            "val_r2": validation["pooled"]["r2"],
        }
        history.append(record)
        print(
            json.dumps(
                {"model": kind, "seed": seed, **record}
            ),
            flush=True,
        )

        if validation["pooled"]["rmse"] < best_val_rmse:
            best_val_rmse = validation["pooled"]["rmse"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError(
            f"{kind} produced no best checkpoint for seed {seed}"
        )

    model.load_state_dict(best_state)
    runtime_seconds = time.perf_counter() - start

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_memory_mb = 0.0

    final_validation = evaluate_model(
        model,
        val_cases,
        stats,
        device,
        torch,
        kind,
    )

    return {
        "model": kind,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "hit_epoch_limit": len(history) == epochs,
        "parameters": parameter_count(model),
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_mb": peak_memory_mb,
        "validation": final_validation,
        "history": history,
        "state_dict": best_state,
    }


def model_config(kind: str, args):
    if kind == "cardignn":
        return CardiGNNConfig(
            in_channels=8,
            hidden_channels=args.hidden,
            out_channels=1,
            layers=args.gnn_layers,
            radius=args.gnn_radius,
            max_num_neighbors=args.gnn_max_neighbors,
            architecture="spatial",
        )

    return CardiGINOConfig(
        in_channels=5,
        out_channels=1,
        hidden_channels=args.hidden,
        spectral_layers=args.gino_spectral_layers,
        grid_size=args.gino_grid_size,
        modes=tuple(args.gino_modes),
        mlp_ratio=args.gino_mlp_ratio,
    )


def run_seed(
    kind,
    seed,
    train_cases,
    val_cases,
    test_cases,
    stats,
    args,
    torch,
    F,
    device,
):
    seed_everything(seed, torch)
    config = model_config(kind, args)

    if kind == "cardignn":
        model = CardiGNN(config).to(device)
    else:
        model = CardiGINO(config).to(device)

    result = train_one(
        model,
        train_cases,
        val_cases,
        stats,
        device,
        torch,
        F,
        kind,
        seed,
        args.epochs,
        args.patience,
        args.lr,
        args.weight_decay,
    )

    result["test"] = evaluate_model(
        model,
        test_cases,
        stats,
        device,
        torch,
        kind,
    )
    result["config"] = config.__dict__

    if args.save_checkpoints:
        checkpoint_dir = args.output / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{kind}_seed_{seed}.pt"
        torch.save(
            {
                "model_state_dict": result["state_dict"],
                "config": config.__dict__,
                "feature_mean": stats[0],
                "feature_std": stats[1],
                "target_mean": stats[2],
                "target_std": stats[3],
                "split_seed": args.split_seed,
                "model_seed": seed,
                "train_indices": args.train_indices,
                "validation_indices": args.validation_indices,
                "test_indices": args.test_indices,
                "source_shard": str(args.shard),
            },
            path,
        )
        result["checkpoint"] = str(path)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result.pop("state_dict", None)
    return result


def summarize(results, baseline):
    grouped = {
        "cardignn": [
            result for result in results if result["model"] == "cardignn"
        ],
        "cardigino": [
            result for result in results if result["model"] == "cardigino"
        ],
    }

    summary = {}
    for kind, rows in grouped.items():
        test_pooled = [row["test"]["pooled"] for row in rows]
        test_macro = [row["test"]["case_macro"] for row in rows]
        val_pooled = [row["validation"]["pooled"] for row in rows]
        epochs = [row["best_epoch"] for row in rows]
        runtimes = [row["runtime_seconds"] for row in rows]

        summary[kind] = {
            "seeds": [row["seed"] for row in rows],
            "test_pooled": aggregate_metrics(test_pooled),
            "test_case_macro": aggregate_metrics(test_macro),
            "validation_pooled": aggregate_metrics(val_pooled),
            "best_epoch": {
                "mean": float(np.mean(epochs)),
                "std": float(np.std(epochs, ddof=1)) if len(epochs) > 1 else 0.0,
                "min": int(min(epochs)),
                "max": int(max(epochs)),
            },
            "runtime_seconds": {
                "mean": float(np.mean(runtimes)),
                "std": float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0,
            },
            "all_hit_epoch_limit": all(
                row["hit_epoch_limit"] for row in rows
            ),
        }

    gnn_test = [
        row["test"]["pooled"] for row in grouped["cardignn"]
    ]
    gino_test = [
        row["test"]["pooled"] for row in grouped["cardigino"]
    ]

    summary["paired_gino_minus_gnn_test_pooled"] = paired_differences(
        gnn_test,
        gino_test,
    )

    summary["baseline"] = baseline
    return summary


def build_report(payload: dict) -> str:
    summary = payload["summary"]
    config = payload["configuration"]
    split = payload["split"]

    lines = [
        "# CardiSim Multi-Seed CardiGNN vs CardiGINO Benchmark",
        "",
        "Longer-training, fixed-split benchmark on the verified DeepCardioSim processed shard.",
        "",
        "## Protocol",
        "",
        f"- Maximum cases: **{config['max_samples']}**",
        f"- Split seed: **{config['split_seed']}**",
        f"- Model seeds: **{config['model_seeds']}**",
        f"- Epoch budget: **{config['epochs']}**",
        f"- Early-stopping patience: **{config['patience']}**",
        "- The case split and training-derived normalization are fixed across all model seeds.",
        "- The held-out test set is evaluated only after selecting the validation-RMSE checkpoint.",
        "- Both pooled-node and case-macro metrics are reported.",
        "- Checkpoints are optional and disabled by default to avoid unnecessary large Git files.",
        "",
        "## Split",
        "",
        f"- Train: **{split['train_cases']} cases**",
        f"- Validation: **{split['validation_cases']} cases**",
        f"- Test: **{split['test_cases']} cases**",
        "",
        "## Test results — pooled nodes",
        "",
        "| Model | MAE (mean ± SD) | RMSE (mean ± SD) | R² (mean ± SD) |",
        "|---|---:|---:|---:|",
    ]

    baseline_row = summary["baseline"]["pooled"]
    lines.append(
        f"| Constant baseline | {baseline_row['mae']:.4f} | "
        f"{baseline_row['rmse']:.4f} | {baseline_row['r2']:.4f} |"
    )

    for label, key in (
        ("CardiGNN", "cardignn"),
        ("CardiGINO", "cardigino"),
    ):
        metrics_row = summary[key]["test_pooled"]
        lines.append(
            f"| {label} | {metrics_row['mae']['mean']:.4f} ± {metrics_row['mae']['std']:.4f} | "
            f"{metrics_row['rmse']['mean']:.4f} ± {metrics_row['rmse']['std']:.4f} | "
            f"{metrics_row['r2']['mean']:.4f} ± {metrics_row['r2']['std']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Test results — case macro",
            "",
            "| Model | MAE (mean ± SD) | RMSE (mean ± SD) | R² (mean ± SD) |",
            "|---|---:|---:|---:|",
        ]
    )

    for label, key in (
        ("CardiGNN", "cardignn"),
        ("CardiGINO", "cardigino"),
    ):
        metrics_row = summary[key]["test_case_macro"]
        lines.append(
            f"| {label} | {metrics_row['mae']['mean']:.4f} ± {metrics_row['mae']['std']:.4f} | "
            f"{metrics_row['rmse']['mean']:.4f} ± {metrics_row['rmse']['std']:.4f} | "
            f"{metrics_row['r2']['mean']:.4f} ± {metrics_row['r2']['std']:.4f} |"
        )

    delta = summary["paired_gino_minus_gnn_test_pooled"]
    lines.extend(
        [
            "",
            "## Paired CardiGINO − CardiGNN test differences",
            "",
            "Negative MAE/RMSE differences and positive R² differences indicate lower error / higher explained variance for CardiGINO in the paired comparison.",
            "",
            "| Metric | Mean delta | SD | Min | Max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in ("mae", "rmse", "r2"):
        row = delta[metric]
        lines.append(
            f"| {metric} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Convergence",
            "",
            f"- CardiGNN all seeds hit epoch limit: **{summary['cardignn']['all_hit_epoch_limit']}**",
            f"- CardiGINO all seeds hit epoch limit: **{summary['cardigino']['all_hit_epoch_limit']}**",
            f"- CardiGNN best epoch mean: **{summary['cardignn']['best_epoch']['mean']:.2f}**",
            f"- CardiGINO best epoch mean: **{summary['cardigino']['best_epoch']['mean']:.2f}**",
            "",
            "## Interpretation boundary",
            "",
            "This benchmark measures seed stability and longer-training behavior on one fixed subset. It does not establish geometry-disjoint generalization, resolution-shift robustness, perturbation robustness, or real-LV performance.",
            "",
            "The next scientific gates remain larger data, geometry-disjoint evaluation, resolution-shift testing, perturbation robustness, and real-LV evaluation.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/multiseed_benchmark"),
    )
    parser.add_argument("--max-samples", type=int, default=128)
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
    parser.add_argument(
        "--gino-modes",
        type=int,
        nargs=3,
        default=[8, 8, 8],
    )
    parser.add_argument(
        "--gino-spectral-layers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--gino-mlp-ratio",
        type=float,
        default=2.0,
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("model seeds must be unique")
    if args.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if args.patience < 1:
        raise ValueError("patience must be >= 1")

    if args.output.exists():
        if not args.overwrite and any(args.output.iterdir()):
            raise SystemExit(
                f"Output directory {args.output} is not empty. "
                "Use --overwrite to replace it."
            )
        if args.overwrite:
            shutil.rmtree(args.output)

    args.output.mkdir(parents=True, exist_ok=True)

    torch, F = torch_modules()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cases = load_cases(args.shard, args.max_samples, torch)
    train_idx, val_idx, test_idx = split_indices(
        len(cases),
        args.split_seed,
    )

    train_raw = [cases[i] for i in train_idx]
    val_raw = [cases[i] for i in val_idx]
    test_raw = [cases[i] for i in test_idx]

    stats = fit_stats(
        train_raw,
        torch,
        max_nodes_per_case=4096,
        seed=args.split_seed,
    )

    train_cases = prepare_cases(train_raw, stats, torch)
    val_cases = prepare_cases(val_raw, stats, torch)
    test_cases = prepare_cases(test_raw, stats, torch)

    args.train_indices = train_idx
    args.validation_indices = val_idx
    args.test_indices = test_idx

    print("================================================")
    print("MULTI-SEED CARDISIM BENCHMARK")
    print("================================================")
    print(f"device: {device}")
    print(f"repository_head: {repo_head()}")
    print(f"split_seed: {args.split_seed}")
    print(f"model_seeds: {args.seeds}")
    print(f"cases: {len(cases)}")
    print(
        "train/validation/test: "
        f"{len(train_cases)}/{len(val_cases)}/{len(test_cases)}"
    )
    print(f"epochs: {args.epochs}")
    print(f"patience: {args.patience}")

    baseline = baseline_scores(test_cases, stats, torch)
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
                train_cases,
                val_cases,
                test_cases,
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

    payload = {
        "benchmark": "cardisim_multiseed_cardignn_cardigino",
        "repository_head": repo_head(),
        "source_shard": str(args.shard),
        "device": str(device),
        "configuration": {
            "max_samples": args.max_samples,
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
            "save_checkpoints": args.save_checkpoints,
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

    json_path = args.output / "multiseed_comparison.json"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    report_path = args.output / "MULTISEED_BENCHMARK_REPORT.md"
    report_path.write_text(
        build_report(payload),
        encoding="utf-8",
    )

    print("================================================")
    print("MULTI-SEED SUMMARY")
    print("================================================")
    print(
        json.dumps(
            payload["summary"],
            indent=2,
        )
    )

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
