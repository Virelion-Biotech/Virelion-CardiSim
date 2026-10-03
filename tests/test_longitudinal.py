import numpy as np

from cardisim.longitudinal import Observation, calibrate_longitudinal
from cardisim.models import N_FEATURES


def test_destructive_timecourse_calibrates_on_subject_means():
    base = np.full(N_FEATURES, 0.4)
    later = base.copy()
    later[0] = 0.6
    observations = [
        Observation("DS", "ST", "s0", "animal1", "sham", 0.0, base, 100),
        Observation("DS", "ST", "s1", "animal1", "mi", 2.0, later, 120),
    ]
    result = calibrate_longitudinal(observations)
    assert result.report.n_transitions == 1
    assert np.isfinite(result.report.rmse)


def test_mixed_subjects_do_not_create_cross_animal_transitions():
    a0 = np.full(N_FEATURES, 0.2)
    a1 = np.full(N_FEATURES, 0.3)
    b0 = np.full(N_FEATURES, 0.8)
    b1 = np.full(N_FEATURES, 0.7)
    observations = [
        Observation("DS", "ST", "a0", "A", "c", 0.0, a0),
        Observation("DS", "ST", "a1", "A", "c", 1.0, a1),
        Observation("DS", "ST", "b0", "B", "c", 0.0, b0),
        Observation("DS", "ST", "b1", "B", "c", 1.0, b1),
    ]
    result = calibrate_longitudinal(observations)
    assert result.report.n_transitions == 2
