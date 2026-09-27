#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix

from hisemble import (
    export_consensus_for_seurat,
    run_hisemble,
    store_consensus_in_anndata,
)


class HisembleApiTests(unittest.TestCase):
    def test_end_to_end_ann_data_workflow(self):
        rng = np.random.default_rng(7)
        adata = ad.AnnData(np.zeros((12, 1)))
        adata.obsm["view_a"] = rng.normal(size=(12, 4))
        adata.obsm["view_b"] = rng.normal(size=(12, 5))
        run_hisemble(
            adata,
            ["view_a", "view_b"],
            metrics=["gaussian", "cosine"],
            k_neighbors=3,
            n_iterations=2,
            resolution=0.5,
            verbose=False,
        )
        self.assertIn("hisemble_clusters", adata.obs)
        self.assertEqual(adata.obsp["connectivities"].shape, (12, 12))

    def test_anndata_and_seurat_export(self):
        identifiers = np.asarray(["cell_1", "cell_2", "cell_3"])
        graph = csr_matrix(
            np.asarray(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            )
        )
        adata = ad.AnnData(np.eye(3), obs={"cell_id": identifiers})
        store_consensus_in_anndata(adata, graph)
        self.assertIn("hisemble_connectivities", adata.obsp)

        with tempfile.TemporaryDirectory() as directory:
            paths = export_consensus_for_seurat(graph, identifiers, Path(directory))
            self.assertTrue(paths["matrix"].exists())
            self.assertTrue(paths["cells"].exists())
            self.assertTrue(paths["metadata"].exists())
            self.assertTrue(paths["readme"].exists())


if __name__ == "__main__":
    unittest.main()
