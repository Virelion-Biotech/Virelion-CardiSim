import numpy as np
import pytest

from cardisim.events import ChallengeEvent, EventSchedule
from cardisim.metrics import area_under_curve, peak_burden
from cardisim import CardiacSimulator, SimulationConfig


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


def test_metrics_are_finite():
    result = CardiacSimulator(SimulationConfig(duration=2, dt=0.5, n_cells=8, seed=1)).run()
    auc = area_under_curve(result, "viability")
    peak = peak_burden(result, "viability")
    assert np.isfinite(auc)
    assert 0 <= peak <= 1
