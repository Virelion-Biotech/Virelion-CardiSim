"""Export durable Colab results without raw data, weights, or manifest self-hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w') as output:
            json.dump(value, output, indent=2, sort_keys=True)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def export(root: Path, destination: Path) -> str:
    root, destination = Path(root), Path(destination)
    inventory, retained = {}, {}
    excluded = {'.pt', '.pth', '.tif', '.tiff', '.png', '.jpg', '.jpeg', '.npy', '.npz'}
    for directory in ['results', 'logs', 'state', 'launcher']:
        for path in (root / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                relative = str(path.relative_to(root))
                if path.suffix.lower() in excluded:
                    retained[relative] = {'sha256': digest(path), 'bytes': path.stat().st_size}
                else:
                    inventory[relative] = digest(path)
    write_json(root / 'RETAINED_ON_DRIVE.json', retained)
    for path in root.iterdir():
        if (path.is_file() and path.name != 'ARTIFACT_HASHES.json'
                and path.resolve() != destination.resolve()
                and path.suffix in {'.json', '.txt', '.log', '.ipynb', '.md'}):
            inventory[path.name] = digest(path)
    write_json(root / 'ARTIFACT_HASHES.json', inventory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination.parent, prefix=destination.name + '.')
    os.close(fd)
    try:
        with zipfile.ZipFile(name, 'w', zipfile.ZIP_DEFLATED) as archive:
            for relative in sorted(inventory):
                archive.write(root / relative, relative)
            archive.write(root / 'ARTIFACT_HASHES.json', 'ARTIFACT_HASHES.json')
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)
    return str(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(export(args.root, args.destination))
