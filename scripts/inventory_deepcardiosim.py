#!/usr/bin/env python3
"""Inventory and optionally download the pinned DeepCardioSim Zenodo record.

The script never assumes filenames or checksums. Zenodo is queried at runtime,
and the returned artifact metadata is written as a machine-readable manifest.
Checksums are verified before a downloaded artifact is moved into place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

RECORD_ID = "17651628"
RECORD_API = f"https://zenodo.org/api/records/{RECORD_ID}"
USER_AGENT = "Virelion-CardiSim/0.3 DeepCardioSim-ingestor"
CHUNK_SIZE = 1024 * 1024


def fetch_record(url: str = RECORD_API) -> dict[str, Any]:
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or "files" not in payload:
        raise ValueError("Zenodo response is missing the expected files collection")
    return payload


def parse_checksum(value: str) -> tuple[str, str]:
    algorithm, separator, digest = value.partition(":")
    if not separator or not algorithm or not digest:
        raise ValueError(f"unsupported checksum format: {value!r}")
    algorithm = algorithm.lower()
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"unsupported checksum algorithm: {algorithm}")
    return algorithm, digest.lower()


def inventory(record: dict[str, Any]) -> dict[str, Any]:
    files = []
    for entry in record["files"]:
        checksum = entry.get("checksum")
        parsed = None
        if checksum:
            algorithm, digest = parse_checksum(checksum)
            parsed = {"algorithm": algorithm, "digest": digest}
        files.append(
            {
                "key": entry.get("key"),
                "size": entry.get("size"),
                "checksum": parsed,
                "links": entry.get("links", {}),
            }
        )
    return {
        "record_id": RECORD_ID,
        "record_url": f"https://zenodo.org/records/{RECORD_ID}",
        "record_version": record.get("version"),
        "revision": record.get("revision"),
        "title": record.get("metadata", {}).get("title"),
        "created": record.get("created"),
        "updated": record.get("updated"),
        "files": files,
    }


def safe_destination(output_dir: Path, key: str) -> Path:
    """Resolve a Zenodo artifact key without allowing path traversal."""
    relative = Path(key)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe Zenodo artifact key: {key!r}")
    destination = (output_dir / relative).resolve()
    root = output_dir.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"unsafe Zenodo artifact key: {key!r}") from exc
    return destination


def download_artifact(url: str, destination: Path, algorithm: str, expected_digest: str) -> None:
    """Download to a temporary sibling, verify, then atomically replace the destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    hasher = hashlib.new(algorithm)
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                hasher.update(chunk)
        actual = hasher.hexdigest().lower()
        if actual != expected_digest.lower():
            raise ValueError(
                f"checksum mismatch for {destination.name}: "
                f"expected {expected_digest}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/deepcardiosim")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/deepcardiosim_manifest.json"),
    )
    parser.add_argument(
        "--download", action="store_true", help="download every Zenodo artifact"
    )
    args = parser.parse_args(argv)

    try:
        record = fetch_record()
        manifest = inventory(record)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        if args.download:
            for artifact in manifest["files"]:
                checksum = artifact["checksum"]
                if not checksum:
                    raise ValueError(
                        f"Zenodo artifact has no checksum: {artifact['key']!r}"
                    )
                url = artifact["links"].get("self") or artifact["links"].get("content")
                if not url:
                    raise ValueError(
                        f"Zenodo artifact has no download URL: {artifact['key']!r}"
                    )
                key = str(artifact["key"])
                destination = safe_destination(args.output, key)
                download_artifact(
                    url,
                    destination,
                    checksum["algorithm"],
                    checksum["digest"],
                )
                print(f"verified {destination}")
        else:
            print(f"inventoried {len(manifest['files'])} Zenodo artifacts")
        return 0
    except Exception as exc:  # pragma: no cover - exercised through CLI/CI
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
