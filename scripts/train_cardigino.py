#!/usr/bin/env python3
"""Bounded native CardiGINO benchmark on a DeepCardioSim processed shard.

The runner is designed for a clean Colab runtime. It uses the same deterministic
case split and training-only normalization policy as the CardiGNN benchmark,
selects the checkpoint by validation RMSE, and evaluates the held-out test set
only after model selection. This is an engineering benchmark, not an exact
reproduction of the published DeepCardioSim training protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from cardisim.cardigino import CardiGINO, CardiGINOConfig


def _torch_modules():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Install the optional GINO dependency: pip install torch") from exc
    return torch, F


def load_shard(path: Path, max_samples: int | None, torch) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, list):
        raise ValueError(f"expected a list of cases in {path}, got {type(payload)!r}")
    cases = payload[:max_samples] if max_samples is not None else payload
    if len(cases) < 6:
        raise ValueError("at least 6 cases are required for train/validation/test splitting")
    return cases


def case_arrays(case: Any, torch):
    a = case["a"] if isinstance(case, dict) else case.a
    pos = case["input_geom"] if isinstance(case, dict) else case.input_geom
    y = case["y"] if isinstance(case, dict) else case.y
    a = torch.as_tensor(a, dtype=torch.float32)
    pos = torch.as_tensor(pos, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32).reshape(pos.shape[0], -1)
    if a.ndim != 2 or a.shape[1] != 5:
        raise ValueError(f"expected a=(N,5), got {tuple(a.shape)}")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"expected input_geom=(N,3), got {tuple(pos.shape)}")
    if y.shape != (pos.shape[0], 1):
        raise ValueError(f"expected y=(N,1), got {tuple(y.shape)}")
    if not bool(torch.isfinite(a).all() and torch.isfinite(pos).all() and torch.isfinite(y).all()):
        raise ValueError("case contains non-finite values")
    return a, pos, y


def split_cases(cases: list[Any], seed: int):
    indices = list(range(len(cases)))
    random.Random(seed).shuffle(indices)
    n = len(indices)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    train_idx = indices[: n - n_val - n_test]
    val_idx = indices[n - n_val - n_test : n - n_test]
    test_idx = indices[n - n_test :]
    return (
        [cases[i] for i in train_idx],
        [cases[i] for i in val_idx],
        [cases[i] for i in test_idx],
        train_idx,
        val_idx,
        test_idx,
    )


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
    x = torch.cat(features, dim=0)
    y = torch.cat(targets, dim=0)
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0, unbiased=False).clamp_min(1e-6)
    return x_mean, x_std, y_mean, y_std


def make_case(case, stats, torch):
    a, pos, y = case_arrays(case, torch)
    x_mean, x_std, y_mean, y_std = stats
    return (
        (a - x_mean[:5]) / x_std[:5],
        pos,
        (y - y_mean) / y_std,
    )


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


def evaluate(model, cases, stats, device, torch):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for a, pos, y in cases:
            a = a.to(device)
            pos = pos.to(device)
            pred = model(a, pos)
            predictions.append(pred.cpu())
            targets.append(y.cpu())
    pred = inverse_target(torch.cat(predictions), stats)
    target = inverse_target(torch.cat(targets), stats)
    return metrics(pred, target)


def baseline(cases, stats, torch):
    target = torch.cat([case[2] for case in cases], dim=0)
    target = inverse_target(target, stats)
    value = stats[2].expand_as(target)
    return metrics(value, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cardigino"))
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--modes", type=int, nargs=3, default=(8, 8, 8))
    parser.add_argument("--spectral-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12130875)
    args = parser.parse_args()

    torch, F = _torch_modules()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = load_shard(args.shard, args.max_samples, torch)
    train_raw, val_raw, test_raw, train_idx, val_idx, test_idx = split_cases(cases, args.seed)
    stats = fit_stats(train_raw, torch, max_nodes_per_case=4096, seed=args.seed)
    train_cases = [make_case(c, stats, torch) for c in train_raw]
    val_cases = [make_case(c, stats, torch) for c in val_raw]
    test_cases = [make_case(c, stats, torch) for c in test_raw]

    config = CardiGINOConfig(
        hidden_channels=args.hidden,
        grid_size=args.grid_size,
        modes=tuple(args.modes),
        spectral_layers=args.spectral_layers,
    )
    model = CardiGINO(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    baseline_val = baseline(val_cases, stats, torch)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for a, pos, y in train_cases:
            a = a.to(device)
            pos = pos.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(a, pos)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= max(len(train_cases), 1)
        val_score = evaluate(model, val_cases, stats, device, torch)
        record = {
            "epoch": epoch,
            "train_mse_normalized": train_loss,
            "val_mae": val_score["mae"],
            "val_rmse": val_score["rmse"],
            "val_r2": val_score["r2"],
        }
        history.append(record)
        print(json.dumps(record))
        if val_score["rmse"] < best_val:
            best_val = val_score["rmse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("no best checkpoint was recorded")
    model.load_state_dict(best_state)
    final_test = evaluate(model, test_cases, stats, device, torch)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "cardigino.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "feature_mean": stats[0],
            "feature_std": stats[1],
            "target_mean": stats[2],
            "target_std": stats[3],
            "seed": args.seed,
            "train_cases": len(train_cases),
            "validation_cases": len(val_cases),
            "test_cases": len(test_cases),
            "train_indices": train_idx,
            "validation_indices": val_idx,
            "test_indices": test_idx,
            "best_epoch": best_epoch,
            "source_shard": str(args.shard),
        },
        checkpoint,
    )
    report = {
        "device": str(device),
        "source_shard": str(args.shard),
        "max_samples": args.max_samples,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "train_cases": len(train_cases),
        "validation_cases": len(val_cases),
        "test_cases": len(test_cases),
        "baseline_validation": baseline_val,
        "final_test": final_test,
        "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline_validation": baseline_val, "final_test": final_test}))
    print(f"saved {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
