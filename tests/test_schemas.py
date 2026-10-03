import json
from pathlib import Path

from jsonschema import Draft202012Validator

from cardisim import CardiacSimulator, SimulationConfig, uncalibrated_cdt_prior


ROOT = Path(__file__).resolve().parents[1]


def _schema(name: str) -> dict:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def test_simulation_result_schema_matches_json_output(tmp_path):
    result = CardiacSimulator(
        SimulationConfig(duration=1.0, dt=0.25, n_cells=4, seed=3, process_noise=0.0)
    ).run()
    output = tmp_path / "result.json"
    result.to_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(_schema("simulation_result.schema.json"))
    Draft202012Validator(_schema("simulation_result.schema.json")).validate(payload)


def test_phenotype_to_cdt_profile_schema_matches_serialization(tmp_path):
    profile = uncalibrated_cdt_prior()
    output = tmp_path / "profile.json"
    profile.save_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(_schema("phenotype_to_cdt_profile.schema.json"))
    Draft202012Validator(_schema("phenotype_to_cdt_profile.schema.json")).validate(payload)
