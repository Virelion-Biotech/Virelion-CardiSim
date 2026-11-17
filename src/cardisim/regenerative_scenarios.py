"""Deterministic synthetic cardiac-regeneration/genotype challenge trajectories.

All axes are normalized latent challenge variables, not biomarkers, fitted biology,
patient predictions, or expected effects of named therapies.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


SCENARIO_VERSION = "1.0.0"
GENERATOR_VERSION = "cardisim-regenerative-scenarios/1.0.0"
SCENARIOS = {
    "post_mi_transient_proliferation": (
        "Post-MI injury with a transient MYC–Cyclin-T1-like proliferation pulse "
        "and independent safety/remodelling axes"
    ),
    "pln_r14del_calcium": (
        "PLN-R14del-like calcium-handling challenge with a synthetic "
        "target-modulation input"
    ),
    "bag3_dcm_proteostasis": (
        "BAG3-associated DCM-like proteostasis and contractility challenge"
    ),
    "pkp2_arrhythmogenic": (
        "PKP2-associated arrhythmogenic-cardiomyopathy-like desmosomal "
        "and electrical challenge"
    ),
    "lamp2_danon_autophagy": (
        "LAMP2/Danon-like autophagic-lysosomal dysfunction and "
        "hypertrophic-remodelling challenge"
    ),
    "immune_resolution_ifng": (
        "Synthetic IFN-gamma/macrophage/matrix-remodelling resolution challenge"
    ),
}


@dataclass(frozen=True)
class RegenerativeScenarioResult:
    scenario_id: str
    time: np.ndarray
    axes: dict[str, np.ndarray]
    manifest: dict[str, Any]

    def mean_trajectory(self) -> dict[str, list[float]]:
        return {
            name: values.mean(axis=0).tolist()
            for name, values in self.axes.items()
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.manifest,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode())
        digest.update(np.ascontiguousarray(self.time).tobytes())
        for axis_name in sorted(self.axes):
            digest.update(axis_name.encode())
            digest.update(
                np.ascontiguousarray(self.axes[axis_name]).tobytes()
            )
        return digest.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "time_days": self.time.tolist(),
            "axes": {
                name: values.tolist()
                for name, values in self.axes.items()
            },
            "manifest": self.manifest,
            "fingerprint": self.fingerprint(),
        }


def _grid(duration: float, dt: float) -> np.ndarray:
    if (
        not np.isfinite(duration)
        or not np.isfinite(dt)
        or duration <= 0
        or dt <= 0
        or duration < dt
    ):
        raise ValueError(
            "duration and dt must be finite, positive, and duration >= dt"
        )
    n_steps = int(np.floor(duration / dt + 1e-12))
    time = np.arange(n_steps + 1, dtype=float) * dt
    if time[-1] < duration - 1e-10:
        time = np.append(time, duration)
    else:
        time[-1] = duration
    return time


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


def _clip(value: np.ndarray) -> np.ndarray:
    return np.clip(value, 0.0, 1.0)


def simulate_regenerative_scenario(
    scenario_id: str,
    *,
    duration: float = 28.0,
    dt: float = 0.25,
    seed: int = 7,
    n_samples: int = 32,
    heterogeneity: float = 0.015,
) -> RegenerativeScenarioResult:
    """Generate a seeded cohort of bounded, explicitly synthetic trajectories."""
    scenario_name = scenario_id.strip().lower()
    if scenario_name not in SCENARIOS:
        raise KeyError(
            f"unknown scenario {scenario_id!r}; choose from {sorted(SCENARIOS)}"
        )
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if not np.isfinite(heterogeneity) or heterogeneity < 0:
        raise ValueError("heterogeneity must be finite and non-negative")

    time = _grid(float(duration), float(dt))
    rng = np.random.default_rng(seed)

    # Static per-sample offsets preserve seeded reproducibility without
    # timestep-dependent noise.
    def sample_offsets(scale: float = 1.0) -> np.ndarray:
        return rng.normal(
            0.0,
            heterogeneity * scale,
            size=(n_samples, 1),
        )

    t = time[None, :]
    axes: dict[str, np.ndarray] = {}

    if scenario_name == "post_mi_transient_proliferation":
        pulse = np.exp(-0.5 * ((t - 4.0) / 1.5) ** 2)
        increments = np.r_[0.0, np.diff(time)]
        cumulative = np.cumsum(increments * pulse[0, :])
        cumulative /= max(float(cumulative[-1]), 1e-12)
        axes = {
            "cell_cycle_entry": _clip(0.08 + 0.78 * pulse + sample_offsets()),
            "cell_number_proxy": _clip(
                0.48 + 0.13 * cumulative[None, :] + sample_offsets(0.5)
            ),
            "contractility_latent": _clip(
                0.38 + 0.18 * (1 - np.exp(-t / 18)) + sample_offsets()
            ),
            "fibrosis_latent": _clip(
                0.52 - 0.07 * (1 - np.exp(-t / 35)) + sample_offsets()
            ),
            "arrhythmia_risk_latent": _clip(
                0.16 + 0.58 * pulse + sample_offsets()
            ),
            "off_target_proliferation_latent": _clip(
                0.06 + 0.42 * pulse + sample_offsets()
            ),
        }
    elif scenario_name == "pln_r14del_calcium":
        modulation = _sigmoid((t - 7.0) / 1.8)
        axes = {
            "calcium_handling_latent": _clip(
                0.28 + 0.35 * modulation + sample_offsets()
            ),
            "target_modulation_latent": _clip(
                0.05 + 0.72 * modulation + sample_offsets()
            ),
            "arrhythmia_risk_latent": _clip(
                0.68 - 0.30 * modulation + sample_offsets()
            ),
            "contractility_latent": _clip(
                0.42 + 0.10 * modulation + sample_offsets()
            ),
        }
    elif scenario_name == "bag3_dcm_proteostasis":
        decline = 1 - np.exp(-t / 35)
        axes = {
            "proteostasis_latent": _clip(
                0.62 - 0.35 * decline + sample_offsets()
            ),
            "sarcomeric_integrity_latent": _clip(
                0.66 - 0.28 * decline + sample_offsets()
            ),
            "contractility_latent": _clip(
                0.62 - 0.30 * decline + sample_offsets()
            ),
            "hypertrophy_latent": _clip(
                0.25 + 0.42 * decline + sample_offsets()
            ),
        }
    elif scenario_name == "pkp2_arrhythmogenic":
        progression = t / max(duration, 1e-12)
        axes = {
            "desmosomal_integrity_latent": _clip(
                0.72 - 0.42 * progression + sample_offsets()
            ),
            "electrophysiology_instability_latent": _clip(
                0.18 + 0.56 * progression + sample_offsets()
            ),
            "arrhythmia_risk_latent": _clip(
                0.20 + 0.62 * progression + sample_offsets()
            ),
            "fibrosis_latent": _clip(
                0.12 + 0.30 * progression + sample_offsets()
            ),
        }
    elif scenario_name == "lamp2_danon_autophagy":
        impairment = 1 - np.exp(-t / 24)
        axes = {
            "autophagic_flux_latent": _clip(
                0.68 - 0.42 * impairment + sample_offsets()
            ),
            "lysosomal_function_latent": _clip(
                0.70 - 0.40 * impairment + sample_offsets()
            ),
            "hypertrophy_latent": _clip(
                0.22 + 0.48 * impairment + sample_offsets()
            ),
            "contractility_latent": _clip(
                0.62 - 0.16 * impairment + sample_offsets()
            ),
        }
    else:
        ifng = 0.72 * np.exp(-t / 9) + 0.08
        macrophage = 0.18 + 0.58 * np.exp(-((t - 4) / 8) ** 2)
        matrix = 0.62 - 0.25 * (1 - np.exp(-t / 20))
        axes = {
            "ifng_signalling_latent": _clip(ifng + sample_offsets()),
            "macrophage_activity_latent": _clip(
                macrophage + sample_offsets()
            ),
            "matrix_remodelling_latent": _clip(matrix + sample_offsets()),
            "fibrosis_latent": _clip(
                0.55 - 0.16 * (1 - np.exp(-t / 22)) + sample_offsets()
            ),
        }

    for name, values in axes.items():
        if values.shape != (n_samples, len(time)):
            raise FloatingPointError(
                f"scenario axis {name} violated output shape contract"
            )
        if not np.isfinite(values).all():
            raise FloatingPointError(
                f"scenario axis {name} contains non-finite values"
            )
        if np.any((values < 0) | (values > 1)):
            raise FloatingPointError(
                f"scenario axis {name} is outside [0, 1]"
            )

    manifest = {
        "schema_version": "1.0",
        "scenario_version": SCENARIO_VERSION,
        "generator_version": GENERATOR_VERSION,
        "scenario_id": scenario_name,
        "description": SCENARIOS[scenario_name],
        "evidence_state": "synthetic",
        "seed": int(seed),
        "n_samples": n_samples,
        "duration_days": float(duration),
        "dt_days": float(dt),
        "heterogeneity_sd": float(heterogeneity),
        "axes": sorted(axes),
        "axis_interpretation": (
            "normalized latent challenge variables; not measured biomarkers"
        ),
        "calibration_status": "uncalibrated_synthetic",
        "clinical_or_therapeutic_efficacy_inference": False,
    }
    return RegenerativeScenarioResult(
        scenario_name,
        time,
        axes,
        manifest,
    )


def timestep_sensitivity(
    scenario_id: str,
    *,
    duration: float = 28.0,
    dt: float = 0.5,
    seed: int = 7,
    n_samples: int = 32,
) -> dict[str, Any]:
    """Compare mean latent trajectories at dt and dt/2 by interpolation."""
    coarse = simulate_regenerative_scenario(
        scenario_id,
        duration=duration,
        dt=dt,
        seed=seed,
        n_samples=n_samples,
        heterogeneity=0.0,
    )
    fine = simulate_regenerative_scenario(
        scenario_id,
        duration=duration,
        dt=dt / 2,
        seed=seed,
        n_samples=n_samples,
        heterogeneity=0.0,
    )
    errors = {}
    for axis_name in coarse.axes:
        interpolated = np.interp(
            coarse.time,
            fine.time,
            fine.axes[axis_name].mean(axis=0),
        )
        errors[axis_name] = float(
            np.sqrt(
                np.mean(
                    (
                        coarse.axes[axis_name].mean(axis=0)
                        - interpolated
                    )
                    ** 2
                )
            )
        )
    return {
        "scenario_id": scenario_id,
        "dt_coarse": dt,
        "dt_fine": dt / 2,
        "rmse_by_axis": errors,
        "max_rmse": max(errors.values(), default=0.0),
        "interpretation": (
            "numerical discretization comparison only; not biological validation"
        ),
    }
