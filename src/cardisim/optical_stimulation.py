"""Minimal optical-stimulation hooks for CardiSim.

This module deliberately separates illumination scheduling from opsin kinetics. It
implements the shared Virelion optical-stimulation metadata contract and a generic
conductance-current helper, but it does not claim a calibrated or validated opsin
model.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

OPTICAL_STIMULATION_SCHEMA_VERSION = "virelion.optical-stimulation/1.0.0"


@dataclass(frozen=True)
class OpticalStimulationProtocol:
    protocol_id: str
    modality: str
    wavelength_nm: float
    irradiance_mw_mm2: float
    pulse_width_ms: float
    frequency_hz: float
    start_ms: float
    duration_ms: float
    control_mode: str = "open_loop"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OpticalStimulationProtocol":
        if payload.get("schema_version") != OPTICAL_STIMULATION_SCHEMA_VERSION:
            raise ValueError("unsupported optical stimulation schema_version")
        light = payload.get("light") or {}
        timing = payload.get("timing") or {}
        control = payload.get("control") or {}
        protocol = cls(
            protocol_id=str(payload.get("protocol_id", "")).strip(),
            modality=str(payload.get("modality", "")).strip(),
            wavelength_nm=float(light["wavelength_nm"]),
            irradiance_mw_mm2=float(light["irradiance_mw_mm2"]),
            pulse_width_ms=float(light["pulse_width_ms"]),
            frequency_hz=float(light["frequency_hz"]),
            start_ms=float(timing["start_ms"]),
            duration_ms=float(timing["duration_ms"]),
            control_mode=str(control.get("mode", "open_loop")),
        )
        protocol.validate()
        return protocol

    def validate(self) -> None:
        numeric = (
            self.wavelength_nm,
            self.irradiance_mw_mm2,
            self.pulse_width_ms,
            self.frequency_hz,
            self.start_ms,
            self.duration_ms,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("optical stimulation values must be finite")
        if not self.protocol_id:
            raise ValueError("protocol_id must be non-empty")
        if self.modality not in {"optogenetic", "optoelectronic"}:
            raise ValueError("unsupported optical stimulation modality")
        if self.control_mode not in {"open_loop", "closed_loop"}:
            raise ValueError("unsupported optical stimulation control mode")
        if self.wavelength_nm <= 0 or self.pulse_width_ms <= 0 or self.duration_ms <= 0:
            raise ValueError("wavelength, pulse width and duration must be positive")
        if self.irradiance_mw_mm2 < 0 or self.frequency_hz < 0 or self.start_ms < 0:
            raise ValueError("irradiance, frequency and start time must be non-negative")


def illumination_active(time_ms: float, protocol: OpticalStimulationProtocol) -> bool:
    """Return whether the protocol schedules light at ``time_ms``.

    Frequency zero is treated as a single pulse. No biological activation kinetics
    are inferred here.
    """
    if not isfinite(time_ms):
        raise ValueError("time_ms must be finite")
    elapsed = time_ms - protocol.start_ms
    if elapsed < 0 or elapsed >= protocol.duration_ms:
        return False
    if protocol.frequency_hz == 0:
        return elapsed < protocol.pulse_width_ms
    period_ms = 1000.0 / protocol.frequency_hz
    return (elapsed % period_ms) < min(protocol.pulse_width_ms, period_ms)


@dataclass(frozen=True)
class OpsinConductance:
    """Generic conductance parameters, independent of a kinetics model."""

    max_conductance_ns: float
    reversal_potential_mv: float

    def __post_init__(self) -> None:
        if not isfinite(self.max_conductance_ns) or self.max_conductance_ns < 0:
            raise ValueError("max_conductance_ns must be finite and non-negative")
        if not isfinite(self.reversal_potential_mv):
            raise ValueError("reversal_potential_mv must be finite")


def opsin_current_pa(
    voltage_mv: float,
    open_fraction: float,
    conductance: OpsinConductance,
) -> float:
    """Return generic opsin current in pA: g[nS] * open_fraction * (V-E)[mV].

    ``open_fraction`` must be supplied by an explicitly selected, separately
    validated kinetics model or experimental estimate. CardiSim does not derive it
    from wavelength or irradiance in this compatibility layer.
    """
    if not isfinite(voltage_mv):
        raise ValueError("voltage_mv must be finite")
    if not isfinite(open_fraction) or not 0.0 <= open_fraction <= 1.0:
        raise ValueError("open_fraction must lie in [0, 1]")
    return conductance.max_conductance_ns * open_fraction * (
        voltage_mv - conductance.reversal_potential_mv
    )
