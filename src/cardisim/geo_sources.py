"""Public GEO source definitions and download helpers.

The helper downloads public *processed* files only when explicitly requested and
records a SHA-256 checksum so calibration artifacts remain reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True)
class GeoSource:
    accession: str
    url: str
    description: str


SOURCES = {
    "GSE185289": GeoSource(
        "GSE185289",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE185289",
        "Pig regenerative/non-regenerative cardiac snRNA-seq time course",
    ),
    "GSE240848": GeoSource(
        "GSE240848",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240848",
        "Rat acute MI and ischemia/reperfusion snRNA-seq",
    ),
    "GSE135310": GeoSource(
        "GSE135310",
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135310",
        "Mouse post-MI inflammatory scRNA-seq time series",
    ),
}


def download(url: str, destination: str | Path) -> str:
    """Download a public resource and return its SHA-256 checksum."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256()
    with urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def source(accession: str) -> GeoSource:
    try:
        return SOURCES[accession.upper()]
    except KeyError as exc:
        raise KeyError(f"unsupported GEO accession: {accession}") from exc
