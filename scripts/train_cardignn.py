#!/usr/bin/env python3
"""Train and evaluate the native CardiGNN on published DeepCardioSim shards.

This is a bounded benchmark runner, not a claim of reproducing the published
training run. It deliberately supports a small sample budget so a Colab GPU
can validate the end-to-end path before a full experiment is attempted.
"""

from __future__ import annotations

import argparse
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
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, list):
        raise ValueError(f"expected a list of cases in {path}, got {type(payload)!r}")
    cases = payload[:max_samples] if max_samples is not None else payload
    if not cases:
        raise ValueError(f"no cases loaded from {path}")
    return cases


def case_to_data(case: Any, torch, Data):
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
    return Data(a=a, input_geom=pos, y=y)


def fit_stats(train_cases: list[Any], torch, max_nodes_per_case: int, seed: int):
    rng = random.Random(seed)
    features: list[Any] = []
    targets: list[Any] = []
    for case in train_cases:
        a = case["a"] if isinstance(case, dict) else case.a
        pos = case["input_geom"] if isinstance(case, dict) else case.input_geom
        y = case["y"] if isinstance(case, dict) else case.y
        a = torch.as_tensor(a, dtype=torch.float32)
        pos = torch.as_tensor(pos, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32).reshape(pos.shape[0], -1)
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
    data.input_geom = (data.input_geom - x_mean[5:]) / x_std[5:]
    data.y = (data.y - y_mean) / y_std
    return data


def inverse_target(values, stats):
    return values * stats[3] + stats[2]


def metrics(pred, target, torch):
    error = pred - target
    mse = float((error.square()).mean().item())
    mae = float(error.abs().mean().item())
    rmse = mse ** 0.5
    ss_res = float(error.square().sum().item())
    centered = target - target.mean()
    ss_tot = float(centered.square().sum().item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cardignn"))
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=12130875)
    args = parser.parse_args()

    torch, F, Data, DataLoader = _torch_modules()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = load_shard(args.shard, args.max_samples)
    split = max(1, int(round(len(cases) * 0.8)))
    split = min(split, len(cases) - 1) if len(cases) > 1 else 1
    train_cases = cases[:split]
    test_cases = cases[split:] if len(cases) > 1 else cases[:1]
    stats = fit_stats(train_cases, torch, max_nodes_per_case=4096, seed=args.seed)
    train_data = [normalize(case_to_data(c, torch, Data), stats) for c in train_cases]
    test_data = [normalize(case_to_data(c, torch, Data), stats) for c in test_cases]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    model = CardiGNN(
        CardiGNNConfig(hidden_channels=args.hidden, layers=4, radius=0.5)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.a, batch.input_geom, batch.batch)
            loss = F.mse_loss(prediction, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= max(len(train_loader), 1)

        model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch.a, batch.input_geom, batch.batch)
                predictions.append(pred.cpu())
                targets.append(batch.y.cpu())
        pred = inverse_target(torch.cat(predictions), stats)
        target = inverse_target(torch.cat(targets), stats)
        score = metrics(pred, target, torch)
        score["epoch"] = epoch
        score["train_mse_normalized"] = train_loss
        history.append(score)
        print(json.dumps(score))

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "cardignn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": CardiGNNConfig(hidden_channels=args.hidden, layers=4, radius=0.5).__dict__,
            "feature_mean": stats[0],
            "feature_std": stats[1],
            "target_mean": stats[2],
            "target_std": stats[3],
            "seed": args.seed,
            "train_cases": len(train_cases),
            "test_cases": len(test_cases),
            "source_shard": str(args.shard),
        },
        checkpoint,
    )
    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "source_shard": str(args.shard),
                "max_samples": args.max_samples,
                "epochs": args.epochs,
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
