"""Phenotype simulation engine and reproducible result container."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dynamics import DynamicsParameters, rk4_step
from .events import EventSchedule
from .models import CardiacState, N_FEATURES, PHENOTYPES, SimulationConfig
from .presets import initial_state

SIMULATION_RESULT_VERSION = "0.4.0"


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass
class SimulationResult:
    """Full population trajectory with reproducibility and provenance metadata."""

    time: np.ndarray
    values: np.ndarray
    cell_ids: np.ndarray
    config: SimulationConfig
    events: tuple[str, ...]
    dynamics_source: str = "default"
    initialization: str = "baseline+heterogeneity"
    event_specs: tuple[dict[str, object], ...] = ()
    initial_state_fingerprint: str = ""
    dynamics_fingerprint: str = ""

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        self.cell_ids = np.asarray(self.cell_ids)
        if self.time.ndim != 1 or self.values.ndim != 3 or self.values.shape[0] != len(self.time):
            raise ValueError("trajectory has invalid shape")
        if self.values.shape[1] != len(self.cell_ids) or self.values.shape[2] != len(PHENOTYPES):
            raise ValueError("trajectory dimensions do not match cell IDs/phenotypes")
        if len(self.time) < 2 or not np.isfinite(self.time).all() or np.any(np.diff(self.time) <= 0):
            raise ValueError("time must be finite and strictly increasing")
        if not np.isfinite(self.values).all():
            raise ValueError("trajectory contains non-finite values")
        if len(np.unique(self.cell_ids)) != len(self.cell_ids):
            raise ValueError("cell_ids must be unique")

    @property
    def final(self) -> CardiacState:
        return CardiacState(self.values[-1], self.cell_ids)

    @property
    def initial(self) -> CardiacState:
        return CardiacState(self.values[0], self.cell_ids)

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 fingerprint of configuration and trajectory."""
        digest = hashlib.sha256()
        digest.update(_canonical_json(self.reproducibility_manifest(include_result_fingerprint=False)))
        digest.update(_array_fingerprint(self.time, self.cell_ids.astype(str), self.values).encode("utf-8"))
        return digest.hexdigest()

    def reproducibility_manifest(self, *, include_result_fingerprint: bool = True) -> dict[str, Any]:
        """Return all metadata required to identify the simulation inputs and outputs."""
        manifest: dict[str, Any] = {
            "result_version": SIMULATION_RESULT_VERSION,
            "config": {
                "duration": self.config.duration,
                "dt": self.config.dt,
                "n_cells": self.config.n_cells,
                "seed": self.config.seed,
                "heterogeneity": self.config.heterogeneity,
                "process_noise": self.config.process_noise,
                "clamp_states": self.config.clamp_states,
            },
            "events": list(self.events),
            "event_specs": [dict(spec) for spec in self.event_specs],
            "dynamics_source": self.dynamics_source,
            "dynamics_fingerprint": self.dynamics_fingerprint,
            "initialization": self.initialization,
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "phenotypes": list(PHENOTYPES),
            "cell_ids": [str(x) for x in self.cell_ids],
        }
        if include_result_fingerprint:
            manifest["fingerprint"] = self.fingerprint()
        return manifest

    def cdt_parameters(self, profile=None) -> dict[str, float]:
        """Translate the final phenotype state through an explicit CDT mapping profile."""
        from .phenotype_to_cdt import uncalibrated_cdt_prior

        mapper = profile or uncalibrated_cdt_prior()
        return mapper.transform(self.final)

    def summary(self) -> dict[str, Any]:
        final_mean = self.final.mean()
        initial_mean = self.initial.mean()
        boundary_fraction = float(np.mean((self.values <= 0) | (self.values >= 1)))
        final_std = {
            name: float(np.std(self.values[-1, :, i], ddof=1)) if len(self.cell_ids) > 1 else 0.0
            for i, name in enumerate(PHENOTYPES)
        }
        return {
            "result_version": SIMULATION_RESULT_VERSION,
            "fingerprint": self.fingerprint(),
            "n_cells": int(len(self.cell_ids)),
            "n_timepoints": int(len(self.time)),
            "duration": float(self.time[-1]),
            "dt_nominal": float(self.config.dt),
            "events": list(self.events),
            "event_specs": [dict(spec) for spec in self.event_specs],
            "dynamics_source": self.dynamics_source,
            "dynamics_fingerprint": self.dynamics_fingerprint,
            "initialization": self.initialization,
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "initial": initial_mean,
            "final": final_mean,
            "final_population_std": final_std,
            "delta": {k: final_mean[k] - initial_mean[k] for k in PHENOTYPES},
            "boundary_fraction": boundary_fraction,
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
            "version": SIMULATION_RESULT_VERSION,
            "dynamics_source": self.dynamics_source,
            "dynamics_fingerprint": self.dynamics_fingerprint,
            "initialization": self.initialization,
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "config": {
                "duration": self.config.duration,
                "dt": self.config.dt,
                "n_cells": self.config.n_cells,
                "seed": self.config.seed,
                "heterogeneity": self.config.heterogeneity,
                "process_noise": self.config.process_noise,
                "clamp_states": self.config.clamp_states,
            },
            "phenotypes": list(PHENOTYPES),
            "events": list(self.events),
            "event_specs": [dict(spec) for spec in self.event_specs],
            "cell_ids": [str(x) for x in self.cell_ids],
            "time": self.time.tolist(),
            "values": self.values.tolist(),
            "summary": self.summary(),
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class CardiacSimulator:
    """Generate phenotype trajectories using default or empirically calibrated dynamics."""

    def __init__(self, config: SimulationConfig | None = None, dynamics: DynamicsParameters | None = None):
        self.config = config or SimulationConfig()
        self.dynamics = dynamics

    def _default_cell_ids(self) -> np.ndarray:
        return np.array([f"cell_{i:06d}" for i in range(self.config.n_cells)])

    @staticmethod
    def _dynamics_fingerprint(dynamics: DynamicsParameters | None) -> str:
        if dynamics is None:
            return "default"
        return _array_fingerprint(dynamics.intercept, dynamics.state_matrix, dynamics.forcing_matrix)

    def initial_population(
        self,
        rng: np.random.Generator,
        initial: CardiacState | Mapping[str, float] | None = None,
    ) -> tuple[CardiacState, np.ndarray, str]:
        if initial is None:
            base = np.array([initial_state()[name] for name in PHENOTYPES], dtype=float)
            noise = rng.normal(0.0, self.config.heterogeneity, size=(self.config.n_cells, N_FEATURES))
            values = np.clip(base[None, :] + noise, 0.0, 1.0)
            ids = self._default_cell_ids()
            return CardiacState(values, ids), ids, "baseline+heterogeneity"
        if isinstance(initial, CardiacState):
            if len(initial.values) != self.config.n_cells:
                raise ValueError("provided CardiacState cell count must equal config.n_cells")
            ids = initial.cell_ids.copy() if initial.cell_ids is not None else self._default_cell_ids()
            state = CardiacState(initial.values.copy(), ids)
            state.validate_bounds()
            return state, ids, "provided-state"
        missing = set(PHENOTYPES) - set(initial)
        extra = set(initial) - set(PHENOTYPES)
        if missing or extra:
            raise ValueError(f"invalid initial phenotype mapping; missing={sorted(missing)}, extra={sorted(extra)}")
        base = np.asarray([float(initial[name]) for name in PHENOTYPES], dtype=float)
        if not np.isfinite(base).all() or np.any((base < 0) | (base > 1)):
            raise ValueError("initial phenotype mapping must be finite and bounded in [0, 1]")
        noise = rng.normal(0.0, self.config.heterogeneity, size=(self.config.n_cells, N_FEATURES))
        values = np.clip(base[None, :] + noise, 0.0, 1.0)
        ids = self._default_cell_ids()
        return CardiacState(values, ids), ids, "custom-mean+heterogeneity"

    def run(
        self,
        schedule: EventSchedule | None = None,
        *,
        initial: CardiacState | Mapping[str, float] | None = None,
    ) -> SimulationResult:
        schedule = schedule or EventSchedule()
        rng = np.random.default_rng(self.config.seed)
        times = self.config.time
        state, cell_ids, initialization = self.initial_population(rng, initial)
        initial_fingerprint = _array_fingerprint(state.values, cell_ids.astype(str))
        trajectory = np.empty((len(times), self.config.n_cells, N_FEATURES), dtype=float)
        trajectory[0] = state.values
        for i in range(1, len(times)):
            t = float(times[i - 1])
            step = float(times[i] - times[i - 1])
            state.values[:] = rk4_step(state.values, t, step, schedule.forcing, self.dynamics)
            if self.config.process_noise:
                state.values[:] += rng.normal(0.0, self.config.process_noise * np.sqrt(step), state.values.shape)
            if self.config.clamp_states:
                state.values[:] = np.clip(state.values, 0.0, 1.0)
            if not np.isfinite(state.values).all():
                raise FloatingPointError(f"non-finite state at t={times[i]}")
            trajectory[i] = state.values
        return SimulationResult(
            times,
            trajectory,
            cell_ids,
            self.config,
            tuple(schedule.names()),
            self.dynamics.source if self.dynamics else "default",
            initialization,
            tuple(schedule.specs()),
            initial_fingerprint,
            self._dynamics_fingerprint(self.dynamics),
        )


def maturity_score(state: CardiacState) -> float:
    idx = [state.values[:, i].mean() for i in [0, 1, 2, 3, 4, 11]]
    return float(np.mean(idx))


def health_score(state: CardiacState) -> float:
    positive = state.values[:, [1, 2, 3, 4, 8, 9, 11]].mean()
    burden = state.values[:, [6, 7, 10]].mean()
    return float(np.clip(positive - 0.55 * burden, 0.0, 1.0))
