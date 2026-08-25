"""Regression tests for the paper-aligned ELA feature extractor."""

import unittest

import numpy as np

from RL.shared.ELA.ELA import ELA_FEATURE_NAMES, compute_ela_features


class PaperELAFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.x = rng.uniform(-5.0, 5.0, size=(100, 5))
        cls.y = np.sum(cls.x**2, axis=1)

    def test_feature_order_dimension_and_determinism(self):
        first = compute_ela_features(self.x, self.y, seed=42)
        second = compute_ela_features(self.x, self.y, seed=42)

        self.assertEqual(
            ELA_FEATURE_NAMES,
            (
                "ela_meta.lin_simple.intercept",
                "ela_meta.quad_simple.adj_r2",
                "ela_meta.lin_w_interact.adj_r2",
                "ic.m0",
                "ic.h_max",
                "ic.eps_ratio",
                "nbc.nn_nb.mean_ratio",
                "nbc.dist_ratio.coeff_var",
                "ela_distr.number_of_peaks",
            ),
        )
        self.assertEqual(first.shape, (9,))
        self.assertTrue(np.all(np.isfinite(first)))
        np.testing.assert_array_equal(first, second)

    def test_rejects_too_few_samples(self):
        with self.assertRaisesRegex(ValueError, "At least 21 samples"):
            compute_ela_features(self.x[:20], self.y[:20])

    def test_rejects_constant_objective_values(self):
        with self.assertRaisesRegex(ValueError, "all objective values are equal"):
            compute_ela_features(self.x, np.ones(self.x.shape[0]))

    def test_warns_for_underdetermined_interaction_model(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(100, 20))
        y = np.sum(x**2, axis=1)
        with self.assertWarnsRegex(RuntimeWarning, "underdetermined"):
            features = compute_ela_features(x, y, seed=42)
        self.assertEqual(features.shape, (9,))


if __name__ == "__main__":
    unittest.main()
