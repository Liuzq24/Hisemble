"""Exercise order invariance in the core distributed to package users."""
import itertools
import unittest

import numpy as np
from scipy.sparse import csr_matrix
from hisemble.core import hisemble_fusion


class ProductionCoreOrderTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.graphs = []
        for _ in range(3):
            values = rng.uniform(0.05, 1.0, size=(8, 8))
            values = (values + values.T) / 2
            np.fill_diagonal(values, 0)
            self.graphs.append(csr_matrix(values))

    def check_all_orders(self, iterations):
        reference = hisemble_fusion(self.graphs, k_neighbors=2,
                                    n_iterations=iterations, verbose=False)
        for order in itertools.permutations(range(3)):
            with self.subTest(order=order):
                actual = hisemble_fusion([self.graphs[i] for i in order],
                                        k_neighbors=2, n_iterations=iterations,
                                        verbose=False)
                difference = (reference - actual).tocsr()
                error = 0 if difference.nnz == 0 else np.abs(difference.data).max()
                self.assertLessEqual(error, 1e-12)

    def test_diffusion_all_six_orders(self):
        self.check_all_orders(iterations=4)

    def test_average_all_six_orders(self):
        self.check_all_orders(iterations=0)


if __name__ == '__main__':
    unittest.main()
