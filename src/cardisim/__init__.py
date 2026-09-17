"""Public API for Virelion-CardiSim."""

from .atlas_bridge import calibrate_atlas_export, validate_atlas_manifest
from .calibration import (
    BootstrapParameterEnsemble,
    CalibrationResult,
    EmpiricalTrajectory,
    SubjectHoldoutCalibration,
    SubjectHoldoutReport,
    bootstrap_calibrate,
    calibrate,
    calibrate_subject_holdout,
    calibration_from_csv,
    load_long_csv,
)
from .cohort_calibration import CohortCalibrationResult, CohortObservation, calibrate_cohort
from .deepcardiosim import DEEPCARDIOSIM_REFERENCE, ReferenceRecord, deepcardiosim_reference
from .deepcardiosim_data import DeepCardioSimSample
from .dynamics import DEFAULT_PARAMETERS, DynamicsParameters
from .events import ChallengeEvent, EventSchedule
from .geo10x import sparse_module_scores
from .geo_sources import SOURCES, GeoSource, download
from .geometry_reference import EPPreprocessor, UnitGaussianNormalizer
from .longitudinal import LongitudinalCalibrationReport, LongitudinalCalibrationResult, Observation, calibrate_longitudinal
from .marker_modules import PROXY_MODULES
from .models import PHENOTYPES, CardiacState, SimulationConfig
from .phenotype_to_cdt import (
    CDT_PARAMETER_NAMES,
    CDTParameterRule,
    MappingFitReport,
    PhenotypeToCDTProfile,
    fit_phenotype_to_cdt,
    load_profile,
    profile_from_dict,
    uncalibrated_cdt_prior,
    validate_phenotype_to_cdt,
)
from .presets import population_preset
from .proxy_targets import mean_target, module_scores
from .simulate import SIMULATION_RESULT_VERSION, CardiacSimulator, SimulationResult, health_score, maturity_score
from .target_derivation import derive_targets, write_long_targets
from .uncertainty import PhenotypeSpread, ensemble_final_means, state_spread
from .validation import ConvergenceReport, DynamicsStabilityReport, TrajectoryDiagnostics, diagnose_result, dynamics_stability, timestep_convergence

__all__ = [
    "BootstrapParameterEnsemble",
    "CDT_PARAMETER_NAMES",
    "DEEPCARDIOSIM_REFERENCE",
    "DEFAULT_PARAMETERS",
    "PHENOTYPES",
    "PROXY_MODULES",
    "SOURCES",
    "SIMULATION_RESULT_VERSION",
    "CalibrationResult",
    "CardiacSimulator",
    "CardiacState",
    "ChallengeEvent",
    "CDTParameterRule",
    "CohortCalibrationResult",
    "CohortObservation",
    "ConvergenceReport",
    "DeepCardioSimSample",
    "DynamicsParameters",
    "DynamicsStabilityReport",
    "EmpiricalTrajectory",
    "EventSchedule",
    "EPPreprocessor",
    "GeoSource",
    "LongitudinalCalibrationReport",
    "LongitudinalCalibrationResult",
    "MappingFitReport",
    "Observation",
    "PhenotypeSpread",
    "PhenotypeToCDTProfile",
    "ReferenceRecord",
    "SimulationConfig",
    "SimulationResult",
    "SubjectHoldoutCalibration",
    "SubjectHoldoutReport",
    "TrajectoryDiagnostics",
    "UnitGaussianNormalizer",
    "bootstrap_calibrate",
    "calibrate",
    "calibrate_atlas_export",
    "calibrate_cohort",
    "calibrate_longitudinal",
    "calibrate_subject_holdout",
    "calibration_from_csv",
    "deepcardiosim_reference",
    "derive_targets",
    "diagnose_result",
    "download",
    "dynamics_stability",
    "ensemble_final_means",
    "fit_phenotype_to_cdt",
    "health_score",
    "load_profile",
    "mean_target",
    "maturity_score",
    "module_scores",
    "population_preset",
    "profile_from_dict",
    "sparse_module_scores",
    "state_spread",
    "timestep_convergence",
    "uncalibrated_cdt_prior",
    "validate_atlas_manifest",
    "validate_phenotype_to_cdt",
    "write_long_targets",
]

__version__ = "0.4.0"
