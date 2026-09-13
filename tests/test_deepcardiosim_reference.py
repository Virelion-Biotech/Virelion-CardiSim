import hashlib
import json
from pathlib import Path

import numpy as np

from cardisim.geometry_reference import EPPreprocessor
from scripts.inventory_deepcardiosim import inventory, parse_checksum


FIXTURE = Path(__file__).parent / "fixtures" / "deepcardiosim_ep_smoke.json"


def load_fixture():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return (
        np.asarray(payload["coordinates"], dtype=float),
        np.asarray(payload["features"], dtype=float),
        np.asarray(payload["target_activation_time"], dtype=float).reshape(-1, 1),
        tuple(payload["query_resolution"]),
    )


def test_deepcardiosim_fixture_contract():
    coordinates, features, targets, resolution = load_fixture()
    assert coordinates.shape == (4, 3)
    assert features.shape == (4, 5)
    assert targets.shape == (4, 1)

    processor = EPPreprocessor().fit(coordinates, features, targets)
    x_norm, f_norm, y_norm = processor.transform(coordinates, features, targets)

    assert np.isfinite(x_norm).all()
    assert np.isfinite(f_norm).all()
    assert np.isfinite(y_norm).all()
    assert np.allclose(processor.inverse_target(y_norm), targets)

    grid = processor.make_query_grid(coordinates, resolution)
    assert grid.points.shape == (np.prod(resolution), 3)
    assert grid.shape == resolution


def test_reference_manifest_contract():
    manifest = json.loads(
        Path("data/references/deepcardiosim.json").read_text(encoding="utf-8")
    )
    assert manifest["reference"]["commit"] == "a0b271a9aee84fd7b6299602dd4dbf8a3a01698f"
    assert manifest["dataset"]["zenodo"].endswith("17651628")
    assert manifest["integration_policy"]["dataset_bytes_in_git"] is False


def test_zenodo_inventory_parser_uses_authoritative_checksums():
    record = {
        "metadata": {"title": "fixture"},
        "version": "1.0",
        "revision": 1,
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "files": [
            {
                "key": "sample.zip",
                "size": 4,
                "checksum": "md5:098f6bcd4621d373cade4e832627b4f6",
                "links": {"self": "https://example.invalid/sample.zip"},
            }
        ],
    }
    manifest = inventory(record)
    assert manifest["files"][0]["checksum"] == {
        "algorithm": "md5",
        "digest": "098f6bcd4621d373cade4e832627b4f6",
    }
    assert parse_checksum("md5:098f6bcd4621d373cade4e832627b4f6")[0] == "md5"
