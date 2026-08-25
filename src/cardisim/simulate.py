"""Simulation engine and result container."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dynamics import DynamicsParameters, rk4_step
from .events import EventSchedule
from .models import CardiacState, SimulationConfig, PHENOTYPES
from .presets import initial_state


@dataclass
class SimulationResult:
    """Full population trajectory with optional calibrated dynamics metadata."""

    time: np.ndarray
    values: np.ndarray
    cell_ids: np.ndarray
    config: SimulationConfig
    events: tuple[str, ...]
    dynamics_source: str = "default"

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        self.cell_ids = np.asarray(self.cell_ids)
        if self.values.ndim != 3 or self.values.shape[0] != len(self.time):
            raise ValueError("trajectory has invalid shape")
        if self.values.shape[1] != len(self.cell_ids) or self.values.shape[2] != len(PHENOTYPES):
            raise ValueError("trajectory dimensions do not match cell IDs/phenotypes")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("trajectory contains non-finite values")

    @property
    def final(self) -> CardiacState:
        return CardiacState(self.values[-1], self.cell_ids)

    @property
    def initial(self) -> CardiacState:
        return CardiacState(self.values[0], self.cell_ids)

    def mean_trajectory(self) -> dict[str, np.ndarray]:
        means = self.values.mean(axis=1)
        return {name: means[:, i] for i, name in enumerate(PHENOTYPES)}

    def summary(self) -> dict[str, Any]:
        final_mean = self.final.mean()
        initial_mean = self.initial.mean()
        return {
            "n_cells": int(len(self.cell_ids)),
            "n_timepoints": int(len(self.time)),
            "duration": float(self.time[-1]),
            "dt_nominal": float(self.config.dt),
            "events": list(self.events),
            "dynamics_source": self.dynamics_source,
            "initial": initial_mean,
            "final": final_mean,
            "delta": {k: final_mean[k] - initial_mean[k] for k in PHENOTYPES},
            "maturity_score": maturity_score(self.final),
            "cardiac_health_score": health_score(self.final),
        }

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "cell_id", *PHENOTYPES])
            for ti, t in enumerate(self.time):
                for ci, cell_id in enumerate(self.cell_ids):
                    writer.writerow([float(t), str(cell_id), *self.values[ti, ci]])

    def to_json(self, path: str | Path) -> None:
        payload = {
            "version": "0.2.0",
            "dynamics_source": self.dynamics_source,
            "config": {
                "duration": self.config.duration,
                "dt": self.config.dt,
                "n_cells": self.config.n_cells,
                "seed": self.config.seed,
                "heterogeneity": self.config.heterogeneity,
                "process_noise": self.config.process_noise,
            },
            "phenotypes": list(PHENOTYPES),
            "events": list(self.events),
            "cell_ids": [str(x) for x in self.cell_ids],
            "time": self.time.tolist(),
            "values": self.values.tolist(),
            "summary": self.summary(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class CardiacSimulator:
    """Generate synthetic trajectories using default or empirically calibrated dynamics."""

    def __init__(self, config: SimulationConfig | None = None, dynamics: DynamicsParameters | None = None):
        self.config = config or SimulationConfig()
        self.dynamics = dynamics

    def initial_population(self, rng: np.random.Generator) -> tuple[CardiacState, np.ndarray]:
        base = np.array([initial_state()[name] for name in PHENOTYPES], dtype=float)
        noise = rng.normal(0.0, self.config.heterogeneity, size=(self.config.n_cells, len(base)))
        values = np.clip(base[None, :] + noise, 0.0, 1.0)
        cell_ids = np.array([f"cell_{i:06d}" for i in range(self.config.n_cells)])
        return CardiacState(values, cell_ids), cell_ids

    def run(self, schedule: EventSchedule | None = None) -> SimulationResult:
        schedule = schedule or EventSchedule()
        rng = np.random.default_rng(self.config.seed)
        times = self.config.time
        state, cell_ids = self.initial_population(rng)
        trajectory = np.empty((len(times), self.config.n_cells, len(PHENOTYPES)), dtype=float)
        trajectory[0] = state.values
        for i in range(1, len(times)):
            t = float(times[i - 1])
            step = float(times[i] - times[i - 1])
            state.values[:] = rk4_step(state.values, t, step, schedule.forcing, self.dynamics)
            if self.config.process_noise:
                state.values[:] += rng.normal(0.0, self.config.process_noise * np.sqrt(step), state.values.shape)
            if self.config.clamp_states:
                state.values[:] = np.clip(state.values, 0.0, 1.0)
            if not np.all(np.isfinite(state.values)):
                raise FloatingPointError(f"non-finite state at t={times[i]}")
            trajectory[i] = state.values
        return SimulationResult(times, trajectory, cell_ids, self.config, tuple(schedule.names()), self.dynamics.source if self.dynamics else "default")


def maturity_score(state: CardiacState) -> float:
    idx = [state.values[:, i].mean() for i in [0, 1, 2, 3, 4, 11]]
    return float(np.mean(idx))


def health_score(state: CardiacState) -> float:
    positive = state.values[:, [1, 2, 3, 4, 8, 9, 11]].mean()
    burden = state.values[:, [6, 7, 10]].mean()
    return float(np.clip(positive - 0.55 * burden, 0.0, 1.0))
