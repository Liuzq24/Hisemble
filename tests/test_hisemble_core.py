#!/usr/bin/env python3

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from component_ablation import fuse


class ViewOrderInvarianceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.graphs = []
        for _ in range(3):
            values = rng.uniform(0.05, 1.0, size=(8, 8))
            values = (values + values.T) / 2.0
            np.fill_diagonal(values, 0.0)
            self.graphs.append(csr_matrix(values))

    def assert_sparse_close(self, first, second, tolerance=1e-12):
        difference = (first - second).tocsr()
        maximum = 0.0 if difference.nnz == 0 else float(np.abs(difference.data).max())
        self.assertLessEqual(maximum, tolerance)

    def test_diffusion_is_invariant_to_view_order(self):
        reference, _ = fuse(
            self.graphs,
            k_neighbors=2,
            n_iterations=4,
            use_diffusion=True,
            use_dynamic_sparsification=True,
        )
        permuted, _ = fuse(
            [self.graphs[2], self.graphs[0], self.graphs[1]],
            k_neighbors=2,
            n_iterations=4,
            use_diffusion=True,
            use_dynamic_sparsification=True,
        )
        self.assert_sparse_close(reference, permuted)

    def test_direct_average_is_invariant_to_view_order(self):
        reference, _ = fuse(
            self.graphs,
            k_neighbors=2,
            n_iterations=0,
            use_diffusion=False,
            use_dynamic_sparsification=True,
        )
        permuted, _ = fuse(
            [self.graphs[1], self.graphs[2], self.graphs[0]],
            k_neighbors=2,
            n_iterations=0,
            use_diffusion=False,
            use_dynamic_sparsification=True,
        )
        self.assert_sparse_close(reference, permuted)


if __name__ == "__main__":
    unittest.main()
