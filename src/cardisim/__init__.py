"""Public API for Virelion-CardiSim."""

from .atlas_bridge import calibrate_atlas_export, validate_atlas_manifest
from .calibration import (
    CalibrationResult,
    EmpiricalTrajectory,
    calibrate,
    calibration_from_csv,
    load_long_csv,
)
from .cohort_calibration import CohortCalibrationResult, CohortObservation, calibrate_cohort
from .deepcardiosim import DEEPCARDIOSIM_REFERENCE, ReferenceRecord, deepcardiosim_reference
from .deepcardiosim_data import DeepCardioSimSample
from .dynamics import DynamicsParameters
from .events import ChallengeEvent, EventSchedule
from .geo10x import sparse_module_scores
from .geo_sources import SOURCES, GeoSource, download
from .geometry_reference import EPPreprocessor, UnitGaussianNormalizer
from .longitudinal import (
    LongitudinalCalibrationReport,
    LongitudinalCalibrationResult,
    Observation,
    calibrate_longitudinal,
)
from .marker_modules import PROXY_MODULES
from .models import PHENOTYPES, CardiacState, SimulationConfig
from .presets import population_preset
from .proxy_targets import mean_target, module_scores
from .simulate import CardiacSimulator, SimulationResult
from .target_derivation import derive_targets, write_long_targets

__all__ = [
    "DEEPCARDIOSIM_REFERENCE",
    "PHENOTYPES",
    "PROXY_MODULES",
    "SOURCES",
    "CalibrationResult",
    "CardiacSimulator",
    "CardiacState",
    "ChallengeEvent",
    "CohortCalibrationResult",
    "CohortObservation",
    "DeepCardioSimSample",
    "DynamicsParameters",
    "EPPreprocessor",
    "EmpiricalTrajectory",
    "EventSchedule",
    "GeoSource",
    "LongitudinalCalibrationReport",
    "LongitudinalCalibrationResult",
    "Observation",
    "ReferenceRecord",
    "SimulationConfig",
    "SimulationResult",
    "UnitGaussianNormalizer",
    "calibrate",
    "calibrate_atlas_export",
    "calibrate_cohort",
    "calibrate_longitudinal",
    "calibration_from_csv",
    "deepcardiosim_reference",
    "derive_targets",
    "download",
    "load_long_csv",
    "mean_target",
    "module_scores",
    "population_preset",
    "sparse_module_scores",
    "validate_atlas_manifest",
    "write_long_targets",
]

__version__ = "0.3.0"
