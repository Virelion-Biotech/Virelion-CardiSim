"""Reference utilities for external cardiac simulation benchmarks.

This module deliberately does not vendor third-party model code. It provides a
small, dependency-light provenance object that CardiSim can use to record the
external reference, dataset version, and the intended validation contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReferenceRecord:
    """Machine-readable provenance for an external scientific reference."""

    name: str
    repository: str
    repository_ref: str
    license: str
    paper_doi: str
    dataset_url: str
    intended_task: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable provenance record."""
        return asdict(self)


DEEPCARDIOSIM_REFERENCE = ReferenceRecord(
    name="DeepCardioSim",
    repository="https://github.com/ehsanngh/DeepCardioSim",
    repository_ref="main@a0b271a9aee84fd7b6299602dd4dbf8a3a01698f",
    license="MIT",
    paper_doi="10.1038/s41746-026-02399-7",
    dataset_url="https://zenodo.org/records/17651628",
    intended_task="Cardiac left-ventricular activation-time prediction from geometry and electrophysiology inputs",
)


def deepcardiosim_reference() -> ReferenceRecord:
    """Return the pinned DeepCardioSim provenance record."""
    return DEEPCARDIOSIM_REFERENCE
