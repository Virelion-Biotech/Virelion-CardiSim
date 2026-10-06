import pytest

from cardisim.optical_stimulation import (
    OPTICAL_STIMULATION_SCHEMA_VERSION,
    OpticalStimulationProtocol,
    OpsinConductance,
    illumination_active,
    opsin_current_pa,
)


def _payload(frequency_hz=2.0):
    return {
        "schema_version": OPTICAL_STIMULATION_SCHEMA_VERSION,
        "protocol_id": "fixture-optical",
        "modality": "optogenetic",
        "light": {
            "wavelength_nm": 470.0,
            "irradiance_mw_mm2": 0.8,
            "pulse_width_ms": 5.0,
            "frequency_hz": frequency_hz,
        },
        "timing": {"start_ms": 100.0, "duration_ms": 1200.0},
        "control": {"mode": "open_loop"},
    }


def test_protocol_parses_shared_contract_and_schedules_pulses():
    protocol = OpticalStimulationProtocol.from_mapping(_payload())
    assert not illumination_active(99.9, protocol)
    assert illumination_active(100.0, protocol)
    assert illumination_active(104.9, protocol)
    assert not illumination_active(106.0, protocol)
    assert illumination_active(600.0, protocol)
    assert not illumination_active(1300.0, protocol)


def test_zero_frequency_is_a_single_pulse():
    protocol = OpticalStimulationProtocol.from_mapping(_payload(frequency_hz=0.0))
    assert illumination_active(100.0, protocol)
    assert not illumination_active(106.0, protocol)


def test_generic_opsin_current_requires_explicit_open_fraction():
    conductance = OpsinConductance(max_conductance_ns=2.0, reversal_potential_mv=0.0)
    assert opsin_current_pa(-60.0, 0.5, conductance) == pytest.approx(-60.0)
    with pytest.raises(ValueError):
        opsin_current_pa(-60.0, 1.1, conductance)


def test_contract_version_is_rejected_when_unknown():
    payload = _payload()
    payload["schema_version"] = "future-version"
    with pytest.raises(ValueError, match="schema_version"):
        OpticalStimulationProtocol.from_mapping(payload)
