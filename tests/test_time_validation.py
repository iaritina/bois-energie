"""Tests de la validation temporelle a fenetres croissantes."""

import unittest

import pandas as pd

from src.evaluation.time_validation import (
    TIME_FOLDS,
    create_fold_frames,
    recommend_model,
    summarize_results,
)


class TimeValidationTests(unittest.TestCase):
    def test_validation_periods_do_not_overlap_or_use_test(self) -> None:
        frame = pd.DataFrame(
            {
                "annee": list(range(2000, 2022)),
                "region": ["SAVA"] * 22,
                "target": range(22),
            }
        )

        folds = create_fold_frames(frame)
        validation_years: list[int] = []
        for definition, train, validation in folds:
            self.assertLess(train["annee"].max(), validation["annee"].min())
            self.assertLessEqual(validation["annee"].max(), 2021)
            validation_years.extend(validation["annee"].tolist())

        self.assertEqual(len(validation_years), len(set(validation_years)))
        self.assertEqual(len(folds), len(TIME_FOLDS))

    def test_summary_and_recommendation_use_mean_wape(self) -> None:
        results = pd.DataFrame(
            {
                "tache": ["test"] * 6,
                "modele": ["ridge"] * 3 + ["forest"] * 3,
                "wape_pct": [10.0, 20.0, 30.0, 12.0, 13.0, 14.0],
                "mae": [100.0] * 6,
                "rmse": [150.0] * 6,
                "r2": [0.8] * 6,
            }
        )

        summary = summarize_results(results)

        self.assertEqual(recommend_model(summary), "forest")
        forest = summary.loc[summary["modele"] == "forest"].iloc[0]
        self.assertEqual(forest["wape_moyen_pct"], 13.0)


if __name__ == "__main__":
    unittest.main()
