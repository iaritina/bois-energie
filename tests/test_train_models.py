"""Tests des metriques et de la selection des modeles."""

import unittest

import numpy as np

from src.models.train_models import compute_metrics, select_best_model


class TrainModelsTests(unittest.TestCase):
    def test_metrics_for_perfect_predictions(self) -> None:
        metrics = compute_metrics(np.array([100.0, 200.0]), np.array([100.0, 200.0]))

        self.assertEqual(metrics["mae"], 0)
        self.assertEqual(metrics["rmse"], 0)
        self.assertEqual(metrics["wape_pct"], 0)
        self.assertEqual(metrics["r2"], 1)

    def test_wape_uses_total_absolute_error(self) -> None:
        metrics = compute_metrics(np.array([100.0, 200.0]), np.array([90.0, 230.0]))

        self.assertAlmostEqual(metrics["mae"], 20)
        self.assertAlmostEqual(metrics["wape_pct"], 40 / 300 * 100)

    def test_selects_lowest_validation_wape(self) -> None:
        validation = {
            "ridge": {"wape_pct": 15.0, "mae": 100.0},
            "random_forest": {"wape_pct": 10.0, "mae": 120.0},
            "gradient_boosting": {"wape_pct": 12.0, "mae": 90.0},
        }

        self.assertEqual(select_best_model(validation), "random_forest")


if __name__ == "__main__":
    unittest.main()
