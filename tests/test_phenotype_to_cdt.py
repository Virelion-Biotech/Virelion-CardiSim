import numpy as np

from cardisim import (
    CDT_PARAMETER_NAMES,
    CardiacState,
    fit_phenotype_to_cdt,
    load_profile,
    uncalibrated_cdt_prior,
    validate_phenotype_to_cdt,
)
from cardisim.models import N_FEATURES


def test_uncalibrated_prior_is_explicit_and_bounded(tmp_path):
    state = CardiacState(np.full((4, N_FEATURES), 0.5))
    profile = uncalibrated_cdt_prior()
    values = profile.transform(state)
    assert profile.calibration_status == "uncalibrated"
    assert set(values) == set(CDT_PARAMETER_NAMES)
    assert values["apd_max"] >= values["apd_min"]
    path = tmp_path / "profile.json"
    profile.save_json(path)
    restored = load_profile(path)
    assert restored.to_dict() == profile.to_dict()


def test_fit_and_validate_phenotype_to_cdt():
    rng = np.random.default_rng(17)
    x = rng.uniform(0.2, 0.8, size=(80, N_FEATURES))
    weights = np.zeros((N_FEATURES, len(CDT_PARAMETER_NAMES)))
    weights[6, 0] = -0.01
    weights[9, 2] = 0.004
    y = 0.0 + (x - 0.5) @ weights
    bases = np.asarray([0.065, 0.051, 0.048, 0.065, 0.060, 0.300, 240.0, 320.0])
    y += bases[None, :]
    profile, report = fit_phenotype_to_cdt(x, y, regularization=1e-10)
    assert report.n_samples == 80
    assert profile.calibration_status == "empirical-fit"
    held_out = rng.uniform(0.2, 0.8, size=(20, N_FEATURES))
    expected = bases[None, :] + (held_out - 0.5) @ weights
    metrics = validate_phenotype_to_cdt(profile, held_out, expected)
    assert all(np.isfinite(row["rmse"]) for row in metrics.values())
    assert metrics["fibre_speed"]["r2"] > 0.99
