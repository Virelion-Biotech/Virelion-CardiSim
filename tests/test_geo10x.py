from pathlib import Path
import gzip

import numpy as np

from cardisim.geo10x import sparse_module_scores


def test_sparse_module_scores_does_not_require_dense_matrix(tmp_path: Path):
    features = tmp_path / "features.tsv.gz"
    barcodes = tmp_path / "barcodes.tsv.gz"
    matrix = tmp_path / "matrix.mtx.gz"
    with gzip.open(features, "wt", encoding="utf-8") as h:
        h.write("MYH7\tMYH7\tGene Expression\n")
        h.write("TNNT2\tTNNT2\tGene Expression\n")
    with gzip.open(barcodes, "wt", encoding="utf-8") as h:
        h.write("cellA-1\ncellB-1\n")
    with gzip.open(matrix, "wt", encoding="utf-8") as h:
        h.write("%%MatrixMarket matrix coordinate real general\n")
        h.write("2 2 2\n")
        h.write("1 1 100\n")
        h.write("2 2 200\n")

    scores, cells, found = sparse_module_scores(matrix, features, barcodes)
    assert cells == ("cellA-1", "cellB-1")
    assert scores.shape == (2, 12)
    assert set(found["maturity"]) <= {"MYH7", "TNNT2"}
    assert np.all((scores >= 0) & (scores <= 1))
