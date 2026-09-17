"""Composable challenge and intervention events."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .models import FEATURE_INDEX, N_FEATURES

KERNEL_NAMES = ("sin2", "box", "gaussian")


@dataclass(frozen=True)
class ChallengeEvent:
    """A time-localized perturbation applied to phenotype dynamics.

    ``effects`` maps phenotype names to signed forcing strengths. Positive forcing
    increases the state; negative forcing decreases it. The default ``sin2``
    kernel preserves the original smooth zero-at-boundary behavior. ``box`` and
    ``gaussian`` are available for calibrated challenge-response experiments.
    """

    name: str
    onset: float = 0.0
    duration: float = 1.0
    magnitude: float = 1.0
    effects: Mapping[str, float] = field(default_factory=dict)
    recovery: float = 1.0
    kernel: str = "sin2"
    kernel_sigma: float = 0.2

    def __post_init__(self) -> None:
        if not np.isfinite(self.onset) or self.onset < 0:
            raise ValueError("event onset must be finite and non-negative")
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("event duration must be finite and positive")
        if not np.isfinite(self.magnitude):
            raise ValueError("event magnitude must be finite")
        if not np.isfinite(self.recovery) or self.recovery < 0:
            raise ValueError("recovery must be finite and non-negative")
        if self.kernel not in KERNEL_NAMES:
            raise ValueError(f"unknown kernel {self.kernel!r}; choose from {KERNEL_NAMES}")
        if not np.isfinite(self.kernel_sigma) or self.kernel_sigma <= 0:
            raise ValueError("kernel_sigma must be finite and positive")
        unknown = set(self.effects) - set(FEATURE_INDEX)
        if unknown:
            raise ValueError(f"unknown phenotype(s): {sorted(unknown)}")
        if any(not np.isfinite(float(value)) for value in self.effects.values()):
            raise ValueError("event effects must be finite")

    @property
    def end(self) -> float:
        return self.onset + self.duration

    def envelope(self, t: float) -> float:
        if t < self.onset or t > self.end:
            return 0.0
        phase = (t - self.onset) / self.duration
        if self.kernel == "sin2":
            return float(np.sin(np.pi * phase) ** 2)
        if self.kernel == "box":
            return 1.0
        centered = (phase - 0.5) / self.kernel_sigma
        return float(np.exp(-0.5 * centered**2))

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

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple):
            raise TypeError(
                "EventSchedule.events must be a tuple of ChallengeEvent objects; "
                "for a single event, include the trailing comma"
            )
        invalid = [event for event in self.events if not isinstance(event, ChallengeEvent)]
        if invalid:
            raise TypeError("EventSchedule.events must contain only ChallengeEvent objects")

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
