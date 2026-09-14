#!/usr/bin/env python3
"""Train/evaluate native CardiGNN models on published DeepCardioSim shards.

The runner keeps graph construction in physical coordinates, fits all
normalization statistics on training cases only, and selects the final model
using a separate validation set. The held-out test set is evaluated once at
the selected epoch. This remains a bounded engineering benchmark, not a claim
of reproducing the published training protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from cardisim.cardignn import CardiGNN, CardiGNNConfig


def _torch_modules():
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Install the optional GNN dependencies first: "
            "pip install torch torch_geometric"
        ) from exc
    return torch, F, Data, DataLoader


def load_shard(path: Path, max_samples: int | None) -> list[Any]:
    torch, _, _, _ = _torch_modules()
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, list):
        raise ValueError(f"expected a list of cases in {path}, got {type(payload)!r}")
    cases = payload[:max_samples] if max_samples is not None else payload
    if not cases:
        raise ValueError(f"no cases loaded from {path}")
    return cases


def _case_arrays(case: Any, torch):
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


def case_to_data(case: Any, torch, Data):
    a, pos, y = _case_arrays(case, torch)
    return Data(
        a=a,
        input_geom=pos.clone(),
        graph_pos=pos.clone(),
        y=y,
    )


def fit_stats(train_cases: list[Any], torch, max_nodes_per_case: int, seed: int):
    rng = random.Random(seed)
    features: list[Any] = []
    targets: list[Any] = []
    for case in train_cases:
        a, pos, y = _case_arrays(case, torch)
        n = pos.shape[0]
        take = min(n, max_nodes_per_case)
        indices = list(range(n))
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


def normalize(data, stats):
    x_mean, x_std, y_mean, y_std = stats
    data.a = (data.a - x_mean[:5]) / x_std[:5]
    # Keep raw graph_pos for physical-radius graph construction. Normalized
    # coordinates remain in input_geom as model features.
    data.input_geom = (data.input_geom - x_mean[5:]) / x_std[5:]
    data.y = (data.y - y_mean) / y_std
    return data


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


def _split_cases(cases: list[Any], seed: int):
    """Deterministically shuffle cases, then create train/validation/test sets."""

    if len(cases) < 6:
        raise ValueError("at least 6 cases are required for train/validation/test splitting")
    indices = list(range(len(cases)))
    random.Random(seed).shuffle(indices)
    n = len(indices)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    if n - n_test - n_val < 1:
        n_val = 1
        n_test = 1
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


def evaluate(model, loader, stats, device, torch):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(
                batch.a,
                batch.input_geom,
                batch.batch,
                batch.graph_pos,
            )
            predictions.append(pred.cpu())
            targets.append(batch.y.cpu())
    pred = inverse_target(torch.cat(predictions), stats)
    target = inverse_target(torch.cat(targets), stats)
    return metrics(pred, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cardignn"))
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--max-neighbors", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--architecture", choices=("spatial", "sage"), default="spatial")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12130875)
    args = parser.parse_args()

    torch, F, Data, DataLoader = _torch_modules()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = load_shard(args.shard, args.max_samples)
    train_cases, val_cases, test_cases, train_idx, val_idx, test_idx = _split_cases(
        cases, args.seed
    )
    stats = fit_stats(train_cases, torch, max_nodes_per_case=4096, seed=args.seed)
    train_data = [normalize(case_to_data(c, torch, Data), stats) for c in train_cases]
    val_data = [normalize(case_to_data(c, torch, Data), stats) for c in val_cases]
    test_data = [normalize(case_to_data(c, torch, Data), stats) for c in test_cases]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    config = CardiGNNConfig(
        hidden_channels=args.hidden,
        layers=args.layers,
        radius=args.radius,
        max_num_neighbors=args.max_neighbors,
        architecture=args.architecture,
    )
    model = CardiGNN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    baseline_value = stats[2]
    baseline_predictions = baseline_value.expand(sum(d.y.shape[0] for d in val_data), 1)
    baseline_target = inverse_target(torch.cat([d.y for d in val_data]), stats)
    baseline_val = metrics(baseline_value.expand_as(baseline_target), baseline_target)

    history = []
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.a, batch.input_geom, batch.batch, batch.graph_pos)
            loss = F.mse_loss(prediction, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= max(len(train_loader), 1)

        val_score = evaluate(model, val_loader, stats, device, torch)
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
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("no best model state was recorded")
    model.load_state_dict(best_state)
    test_score = evaluate(model, test_loader, stats, device, torch)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "cardignn.pt"
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
        "architecture": args.architecture,
        "source_shard": str(args.shard),
        "max_samples": args.max_samples,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "train_cases": len(train_cases),
        "validation_cases": len(val_cases),
        "test_cases": len(test_cases),
        "baseline_validation": baseline_val,
        "final_test": test_score,
        "history": history,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"baseline_validation": baseline_val, "final_test": test_score}))
    print(f"saved {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
