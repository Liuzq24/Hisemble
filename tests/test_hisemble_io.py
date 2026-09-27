#!/usr/bin/env python3

import unittest

import numpy as np

from hisemble.io import validate_embeddings


class ValidateEmbeddingsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.embeddings = {
            "view_a": rng.normal(size=(20, 4)),
            "view_b": rng.normal(size=(20, 7)),
        }
        self.ids = [f"cell_{index}" for index in range(20)]

    def test_valid_different_dimensions(self):
        reports = validate_embeddings(
            self.embeddings,
            cell_ids_by_view={"view_a": self.ids, "view_b": self.ids},
            outlier_action="ignore",
        )
        self.assertEqual(reports["view_a"].n_cells, 20)
        self.assertEqual(reports["view_b"].n_features, 7)

    def test_nan_rejected(self):
        bad = {name: value.copy() for name, value in self.embeddings.items()}
        bad["view_a"][3, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            validate_embeddings(bad)

    def test_infinite_rejected(self):
        bad = {name: value.copy() for name, value in self.embeddings.items()}
        bad["view_b"][8, 2] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            validate_embeddings(bad)

    def test_cell_count_mismatch_rejected(self):
        bad = dict(self.embeddings)
        bad["view_b"] = bad["view_b"][:-1]
        with self.assertRaisesRegex(ValueError, "Cell-count mismatch"):
            validate_embeddings(bad)

    def test_cell_order_mismatch_rejected(self):
        reversed_ids = list(reversed(self.ids))
        with self.assertRaisesRegex(ValueError, "Cell order differs"):
            validate_embeddings(
                self.embeddings,
                cell_ids_by_view={"view_a": self.ids, "view_b": reversed_ids},
            )

    def test_degenerate_view_rejected(self):
        bad = dict(self.embeddings)
        bad["view_b"] = np.ones((20, 7))
        with self.assertRaisesRegex(ValueError, "degenerate"):
            validate_embeddings(bad)


if __name__ == "__main__":
    unittest.main()
