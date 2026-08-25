"""Public API for Virelion-CardiSim."""

from .events import ChallengeEvent, EventSchedule
from .models import PHENOTYPES, CardiacState, SimulationConfig
from .presets import population_preset
from .simulate import CardiacSimulator, SimulationResult

__all__ = [
    "CardiacSimulator",
    "SimulationConfig",
    "SimulationResult",
    "CardiacState",
    "ChallengeEvent",
    "EventSchedule",
    "PHENOTYPES",
    "population_preset",
]

__version__ = "0.1.0"
