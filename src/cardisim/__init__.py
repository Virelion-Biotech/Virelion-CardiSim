"""Public API for Virelion-CardiSim."""

from .atlas_bridge import calibrate_atlas_export, validate_atlas_manifest
from .calibration import CalibrationResult, EmpiricalTrajectory, calibrate, calibration_from_csv, load_long_csv
from .dynamics import DynamicsParameters
from .events import ChallengeEvent, EventSchedule
from .models import PHENOTYPES, CardiacState, SimulationConfig
from .presets import population_preset
from .simulate import CardiacSimulator, SimulationResult

__all__ = [
    "CardiacSimulator", "SimulationConfig", "SimulationResult", "CardiacState",
    "ChallengeEvent", "EventSchedule", "PHENOTYPES", "population_preset",
    "DynamicsParameters", "EmpiricalTrajectory", "CalibrationResult", "calibrate",
    "load_long_csv", "calibration_from_csv", "validate_atlas_manifest", "calibrate_atlas_export",
]

__version__ = "0.2.0"
