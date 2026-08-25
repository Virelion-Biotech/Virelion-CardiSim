"""Composable challenge and intervention events."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .models import FEATURE_INDEX, N_FEATURES


@dataclass(frozen=True)
class ChallengeEvent:
    """A time-localized perturbation applied to phenotype dynamics.

    ``effects`` maps phenotype names to signed forcing strengths. Positive forcing
    increases the state; negative forcing decreases it. A smooth raised-cosine
    envelope prevents discontinuities at onset/offset.
    """

    name: str
    onset: float = 0.0
    duration: float = 1.0
    magnitude: float = 1.0
    effects: Mapping[str, float] = field(default_factory=dict)
    recovery: float = 1.0

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("event duration must be positive")
        if self.onset < 0:
            raise ValueError("event onset cannot be negative")
        if self.recovery < 0:
            raise ValueError("recovery must be non-negative")
        unknown = set(self.effects) - set(FEATURE_INDEX)
        if unknown:
            raise ValueError(f"unknown phenotype(s): {sorted(unknown)}")

    @property
    def end(self) -> float:
        return self.onset + self.duration

    def envelope(self, t: float) -> float:
        if t < self.onset or t > self.end:
            return 0.0
        phase = (t - self.onset) / self.duration
        return float(np.sin(np.pi * phase) ** 2)

    def forcing(self, t: float) -> np.ndarray:
        vector = np.zeros(N_FEATURES, dtype=float)
        amp = self.magnitude * self.envelope(t)
        for name, value in self.effects.items():
            vector[FEATURE_INDEX[name]] = amp * value * self.recovery
        return vector


@dataclass(frozen=True)
class EventSchedule:
    """Collection of challenge and intervention events."""

    events: tuple[ChallengeEvent, ...] = ()

    def forcing(self, t: float) -> np.ndarray:
        if not self.events:
            return np.zeros(N_FEATURES, dtype=float)
        return np.sum([event.forcing(t) for event in self.events], axis=0)

    def names(self) -> list[str]:
        return [event.name for event in self.events]

    def __iter__(self):
        return iter(self.events)

    def add(self, *events: ChallengeEvent) -> "EventSchedule":
        return EventSchedule(self.events + tuple(events))
