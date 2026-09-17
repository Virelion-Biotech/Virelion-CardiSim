import numpy as np
import pytest

from cardisim import CardiacSimulator, SimulationConfig
from cardisim.events import ChallengeEvent, EventSchedule, KERNEL_NAMES
from cardisim.metrics import area_under_curve, peak_burden


def test_event_envelope_is_zero_outside_window():
    event = ChallengeEvent("x", onset=2, duration=4, effects={"inflammation": 1})
    assert event.envelope(1.9) == 0
    assert event.envelope(6.1) == 0
    assert event.envelope(4.0) == pytest.approx(1.0)


def test_schedule_adds_forcing():
    a = ChallengeEvent("a", effects={"inflammation": 1}, duration=2)
    b = ChallengeEvent("b", effects={"viability": -1}, duration=2)
    forcing = EventSchedule((a, b)).forcing(1.0)
    assert forcing[7] > 0
    assert forcing[9] < 0


def test_explicit_response_kernels():
    assert KERNEL_NAMES == ("sin2", "box", "gaussian")
    box = ChallengeEvent("box", onset=1, duration=2, kernel="box", effects={"inflammation": 1})
    gaussian = ChallengeEvent("gaussian", onset=1, duration=2, kernel="gaussian", kernel_sigma=0.2, effects={"inflammation": 1})
    assert box.envelope(1.0) == pytest.approx(1.0)
    assert box.envelope(1.5) == pytest.approx(1.0)
    assert box.envelope(3.0) == pytest.approx(1.0)
    assert gaussian.envelope(2.0) == pytest.approx(1.0)
    assert gaussian.envelope(1.0) < 1.0


def test_invalid_kernel_and_effects_are_rejected():
    with pytest.raises(ValueError, match="unknown kernel"):
        ChallengeEvent("x", kernel="invalid")
    with pytest.raises(ValueError, match="event effects"):
        ChallengeEvent("x", effects={"inflammation": np.inf})


def test_metrics_are_finite():
    result = CardiacSimulator(SimulationConfig(duration=2, dt=0.5, n_cells=8, seed=1)).run()
    auc = area_under_curve(result, "viability")
    peak = peak_burden(result, "viability")
    assert np.isfinite(auc)
    assert 0 <= peak <= 1
