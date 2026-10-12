#!/usr/bin/env python3
"""Meaningful equivalence and diagnostic checks on small, unsampled fixtures."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz

from vendor_core.core import hisemble_fusion
from vendor_core.graph import build_knn_graph_chunked_cpu
from run_core_convergence import (
    actual_support_change, boundary_gap, instrumented_fusion, source_manifest,
)


class CoreConvergenceTests(unittest.TestCase):
    def test_twenty_iterations_match_submitted_core_exactly(self):
        random = np.random.default_rng(123)
        # Mixed source dtypes and tied duplicated rows cover subtle historical behavior.
        embeddings = [random.normal(size=(90, d)).astype(dtype) for d, dtype in ((12, "float32"), (18, "float64"), (9, "float32"))]
        for array in embeddings:
            array[1:4] = array[0]
        graphs = [build_knn_graph_chunked_cpu(array, 15, metric, chunk_size=17) for array, metric in zip(embeddings, ("gaussian", "gaussian", "cosine"))]
        expected = hisemble_fusion(graphs, k_neighbors=15, n_iterations=20, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            actual, at20, history = instrumented_fusion(graphs, k_neighbors=15, n_iterations=20)
        difference = actual - expected
        self.assertEqual(difference.nnz, 0)
        self.assertEqual((at20 - expected).nnz, 0)
        self.assertEqual(len(history), 60)
        self.assertTrue(all(item["P_relative_fro"] >= 0 for item in history))
        self.assertTrue(all(item["Q_relative_fro"] >= 0 for item in history))
        self.assertTrue(all(abs(item["P_row_sum_min"] - 1) < 1e-6 for item in history))
        self.assertTrue(all(item["P_min_stored_value"] >= 0 for item in history))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):
                tracked, unused, track_history = instrumented_fusion(graphs, k_neighbors=15, n_iterations=21, track_a=True, output_dir=output)
            self.assertEqual((load_npz(output / "consensus_T20.npz") - expected).nnz, 0)
            expected21 = hisemble_fusion(graphs, k_neighbors=15, n_iterations=21, verbose=False)
            self.assertEqual((tracked - expected21).nnz, 0)
            self.assertIsNone(unused)
            self.assertIsNone(track_history[0]["A_relative_fro"])
            self.assertGreaterEqual(track_history[3]["A_relative_fro"], 0)
            self.assertTrue((output / "convergence_history.csv").exists())

    def test_boundary_gap_includes_implicit_zeros_and_ties(self):
        dense = np.array([[9, 5, 3, 1, 0], [4, 4, 4, 0, 0], [8, 2, 0, 0, 0], [7, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=float)
        result = boundary_gap(csr_matrix(dense), 2)
        dense_gaps = np.sort(dense, axis=1)[:, -2] - np.sort(dense, axis=1)[:, -3]
        self.assertEqual(result["boundary_gap_min"], float(dense_gaps.min()))
        self.assertEqual(result["boundary_gap_median"], float(np.median(dense_gaps)))
        self.assertEqual(result["boundary_gap_zero_rows"], 3)
        self.assertEqual(result["rows_with_fewer_than_q_positive_entries"], 2)

    def test_actual_support_ignores_stored_zeros_without_mutation(self):
        old = csr_matrix(([0.0, 1.0], ([0, 1], [1, 2])), shape=(3, 3))
        new = csr_matrix(([2.0, 5.0], ([1, 2], [2, 0])), shape=(3, 3))
        result = actual_support_change(new, old)
        self.assertEqual(result["support_symmetric_difference_nnz"], 1)
        self.assertEqual(result["support_union_nnz"], 2)
        self.assertEqual(result["support_changed_rows"], 1)
        self.assertEqual(old.nnz, 2)
        self.assertEqual(old.data[0], 0)

    def test_vendored_source_hashes_match_submission(self):
        manifest = source_manifest()
        self.assertEqual(set(manifest["sha256"]), {"core.py", "graph.py", "io.py"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
