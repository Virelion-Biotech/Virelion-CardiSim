"""Fetch the public processed resources required for calibration.

This script performs network I/O only when explicitly invoked. It writes files
under a user-selected directory and records SHA-256 checksums.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"
RESOURCES = {
    "GSE240848": {
        "barcodes": f"{BASE}/GSE240nnn/GSE240848/suppl/GSE240848_barcodes.tsv.gz",
        "features": f"{BASE}/GSE240nnn/GSE240848/suppl/GSE240848_features.tsv.gz",
        "matrix": f"{BASE}/GSE240nnn/GSE240848/suppl/GSE240848_matrix.mtx.gz",
    },
    "GSE185289": {
        "seurat": f"{BASE}/GSE185nnn/GSE185289/suppl/GSE185289_Cardiomyocyte_Seurat.Robj.gz",
        "raw": f"{BASE}/GSE185nnn/GSE185289/suppl/GSE185289_RAW.tar",
    },
    "GSE135310": {
        "raw": f"{BASE}/GSE135nnn/GSE135310/suppl/GSE135310_RAW.tar",
    },
}


def download(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(RESOURCES), default="GSE240848")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    checksums: dict[str, str] = {}
    for name, url in RESOURCES[args.dataset].items():
        destination = args.output / args.dataset / Path(url).name
        checksums[name] = download(url, destination)
        print(f"{destination}\t{checksums[name]}")
    checksum_file = args.output / args.dataset / "SHA256SUMS"
    checksum_file.write_text("".join(f"{value}  {key}\n" for key, value in checksums.items()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
