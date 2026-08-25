"""Preset cardiac trajectories and challenge schedules."""
from __future__ import annotations

from .events import ChallengeEvent, EventSchedule
from .models import PHENOTYPES

BASELINE = {
    "maturity": 0.55,
    "contractility": 0.62,
    "calcium_handling": 0.58,
    "electrophysiology": 0.65,
    "metabolism": 0.55,
    "hypertrophy": 0.28,
    "fibrosis": 0.08,
    "inflammation": 0.10,
    "angiogenesis": 0.55,
    "viability": 0.96,
    "oxidative_stress": 0.15,
    "mitochondrial_health": 0.63,
}

PRESETS = {
    "baseline": EventSchedule(),
    "maturation": EventSchedule(
        (
            ChallengeEvent(
                name="developmental_maturation",
                onset=0.0,
                duration=28.0,
                magnitude=0.8,
                effects={
                    "maturity": 0.22,
                    "contractility": 0.12,
                    "calcium_handling": 0.14,
                    "electrophysiology": 0.10,
                    "metabolism": 0.10,
                    "mitochondrial_health": 0.12,
                },
            ),
        )
    ),
    "mi": EventSchedule(
        (
            ChallengeEvent(
                name="myocardial_infarction_like_injury",
                onset=0.0,
                duration=2.0,
                magnitude=1.0,
                effects={
                    "contractility": -0.55,
                    "calcium_handling": -0.30,
                    "electrophysiology": -0.22,
                    "metabolism": -0.28,
                    "hypertrophy": 0.20,
                    "fibrosis": 0.55,
                    "inflammation": 0.70,
                    "angiogenesis": -0.10,
                    "viability": -0.42,
                    "oxidative_stress": 0.62,
                    "mitochondrial_health": -0.45,
                },
                recovery=1.0,
            ),
            ChallengeEvent(
                name="post_injury_remodeling",
                onset=2.0,
                duration=26.0,
                magnitude=0.45,
                effects={
                    "maturity": 0.03,
                    "contractility": 0.05,
                    "fibrosis": 0.10,
                    "inflammation": -0.04,
                    "angiogenesis": 0.04,
                    "oxidative_stress": -0.03,
                    "mitochondrial_health": 0.04,
                },
            ),
        )
    ),
    "hypoxia": EventSchedule(
        (
            ChallengeEvent(
                name="hypoxic_stress",
                onset=0.0,
                duration=3.0,
                magnitude=1.0,
                effects={
                    "contractility": -0.28,
                    "metabolism": -0.32,
                    "viability": -0.20,
                    "oxidative_stress": 0.30,
                    "mitochondrial_health": -0.27,
                    "inflammation": 0.18,
                    "angiogenesis": 0.12,
                },
            )
        )
    ),
    "radiation": EventSchedule(
        (
            ChallengeEvent(
                name="radiation_like_injury",
                onset=0.0,
                duration=1.5,
                magnitude=1.0,
                effects={
                    "viability": -0.18,
                    "oxidative_stress": 0.55,
                    "mitochondrial_health": -0.35,
                    "inflammation": 0.34,
                    "fibrosis": 0.28,
                    "angiogenesis": -0.20,
                    "contractility": -0.15,
                },
            ),
            ChallengeEvent(
                name="late_remodeling",
                onset=7.0,
                duration=21.0,
                magnitude=0.65,
                effects={
                    "fibrosis": 0.18,
                    "inflammation": -0.05,
                    "angiogenesis": 0.04,
                    "mitochondrial_health": 0.03,
                },
            ),
        )
    ),
    "electrotox": EventSchedule(
        (
            ChallengeEvent(
                name="electrophysiology_stress",
                onset=0.0,
                duration=1.0,
                magnitude=1.0,
                effects={
                    "electrophysiology": -0.65,
                    "calcium_handling": -0.25,
                    "contractility": -0.18,
                    "viability": -0.08,
                    "oxidative_stress": 0.10,
                },
            )
        )
    ),
}


def initial_state() -> dict[str, float]:
    """Return a copy of the canonical starting state."""
    return BASELINE.copy()


def population_preset(name: str) -> EventSchedule:
    """Return a named schedule; raises a useful error for unknown presets."""
    key = name.lower().strip()
    if key not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[key]


def preset_names() -> tuple[str, ...]:
    return tuple(sorted(PRESETS))


def validate_presets() -> None:
    """Internal consistency check for packaged presets."""
    assert set(BASELINE) == set(PHENOTYPES)
