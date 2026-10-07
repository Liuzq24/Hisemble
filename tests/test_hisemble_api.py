#!/usr/bin/env python3

import unittest

import anndata as ad
import numpy as np

from hisemble import run_hisemble


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


if __name__ == "__main__":
    unittest.main()
