"""Public API for Virelion-CardiSim."""

from .atlas_bridge import calibrate_atlas_export, validate_atlas_manifest
from .calibration import CalibrationResult, EmpiricalTrajectory, calibrate, calibration_from_csv, load_long_csv
from .dynamics import DynamicsParameters
from .events import ChallengeEvent, EventSchedule
from .geo_sources import SOURCES, GeoSource, download
from .geo10x import sparse_module_scores
from .longitudinal import LongitudinalCalibrationReport, LongitudinalCalibrationResult, Observation, calibrate_longitudinal
from .marker_modules import PROXY_MODULES
from .models import PHENOTYPES, CardiacState, SimulationConfig
from .presets import population_preset
from .proxy_targets import module_scores, mean_target
from .simulate import CardiacSimulator, SimulationResult
from .target_derivation import derive_targets, write_long_targets

__all__ = [
    "CardiacSimulator", "SimulationConfig", "SimulationResult", "CardiacState",
    "ChallengeEvent", "EventSchedule", "PHENOTYPES", "population_preset",
    "DynamicsParameters", "EmpiricalTrajectory", "CalibrationResult", "calibrate",
    "load_long_csv", "calibration_from_csv", "validate_atlas_manifest", "calibrate_atlas_export",
    "GeoSource", "SOURCES", "download", "sparse_module_scores", "PROXY_MODULES",
    "derive_targets", "write_long_targets", "Observation", "LongitudinalCalibrationReport",
    "LongitudinalCalibrationResult", "calibrate_longitudinal", "module_scores", "mean_target",
]

__version__ = "0.3.0"
