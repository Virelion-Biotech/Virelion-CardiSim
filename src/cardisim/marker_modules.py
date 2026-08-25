"""Transparent cross-species cardiac proxy modules for empirical target derivation.

These are *proxy* signatures, not validated clinical biomarkers. They convert
normalized expression matrices into latent phenotype targets that can be fitted
and independently audited.
"""
from __future__ import annotations

PROXY_MODULES: dict[str, tuple[str, ...]] = {
    "maturity": ("MYH7", "TNNI3", "TNNT2", "RYR2", "ATP2A2", "PLN"),
    "contractility": ("TNNT2", "ACTN2", "MYH6", "MYH7", "MYL2"),
    "calcium_handling": ("RYR2", "ATP2A2", "CACNA1C", "PLN", "SLC8A1"),
    "electrophysiology": ("SCN5A", "KCNJ2", "KCNH2", "KCNQ1", "KCND3"),
    "metabolism": ("PPARGC1A", "CPT1B", "ACADM", "HADHA", "COX5A"),
    "hypertrophy": ("NPPA", "NPPB", "MYH7", "ACTA1"),
    "fibrosis": ("COL1A1", "COL3A1", "DCN", "LUM", "TAGLN"),
    "inflammation": ("IL1B", "TNF", "CCL2", "S100A8", "S100A9", "NFKBIA"),
    "angiogenesis": ("KDR", "EMCN", "PECAM1", "ESAM", "ENG", "ANGPT1"),
    "viability": ("BCL2", "BCL2L1", "MCL1"),
    "oxidative_stress": ("HMOX1", "NQO1", "TXNIP", "SOD2", "GPX1"),
    "mitochondrial_health": ("TFAM", "PPARGC1A", "NDUFA1", "COX5A", "ATP5F1E"),
}


def available_fraction(genes: list[str] | tuple[str, ...] | set[str]) -> dict[str, float]:
    present = {g.upper() for g in genes}
    return {
        name: sum(g in present for g in markers) / len(markers)
        for name, markers in PROXY_MODULES.items()
    }
