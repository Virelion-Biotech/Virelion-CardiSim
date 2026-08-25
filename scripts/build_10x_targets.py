"""Build sample-level CardiSim targets from a GEO 10X MatrixMarket triple."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cardisim.geo10x import sparse_module_scores
from cardisim.models import PHENOTYPES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--barcodes", type=Path, required=True)
    parser.add_argument("--sample-map", type=Path, required=True, help="JSON mapping of barcode suffix to sample metadata")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mapping = json.loads(args.sample_map.read_text(encoding="utf-8"))
    scores, barcodes, found = sparse_module_scores(args.matrix, args.features, args.barcodes)

    grouped: dict[str, list[int]] = {}
    for i, barcode in enumerate(barcodes):
        key = barcode.rsplit("-", 1)[-1]
        meta = mapping.get(key)
        if meta is None:
            raise ValueError(f"no sample mapping for barcode suffix -{key}; refusal prevents silent condition assignment")
        grouped.setdefault(meta["sample_id"], []).append(i)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset_id", "study_id", "subject_id", "condition", "time", "sample_id", "n_cells", "phenotype", "value", "module_genes"])
        for sample_id, indices in grouped.items():
            meta = next(mapping[barcodes[i].rsplit("-", 1)[-1]] for i in indices)
            mean = scores[indices].mean(axis=0)
            for phenotype_index, phenotype in enumerate(PHENOTYPES):
                writer.writerow([
                    meta["dataset_id"], meta["study_id"], meta["subject_id"],
                    meta["condition"], meta["time"], sample_id, len(indices),
                    phenotype, float(np.clip(mean[phenotype_index], 0, 1)),
                    ";".join(found[phenotype]),
                ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
