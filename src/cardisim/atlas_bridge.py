from __future__ import annotations

from pathlib import Path
import json

from .calibration import CalibrationResult, calibration_from_csv

ATLAS_MANIFEST_FIELDS = {
    "dataset_id", "study_id", "organism", "modality", "condition",
    "time_unit", "source_accession", "provenance_url"
}

def validate_atlas_manifest(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = sorted(ATLAS_MANIFEST_FIELDS - set(payload))
    if missing:
        raise ValueError(f"Atlas manifest missing fields: {missing}")
    if not payload["dataset_id"] or not payload["study_id"]:
        raise ValueError("dataset_id and study_id must be non-empty")
    return payload

def calibrate_atlas_export(manifest_path: str | Path, observations_path: str | Path, regularization: float = 1e-3) -> CalibrationResult:
    manifest = validate_atlas_manifest(manifest_path)
    return calibration_from_csv(observations_path, str(manifest["dataset_id"]), str(manifest["study_id"]), regularization)
