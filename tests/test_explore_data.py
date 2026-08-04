"""Tests des calculs de l'analyse exploratoire."""

import unittest

import pandas as pd

from src.analysis.explore_data import build_summary, compute_statistics


class ExploreDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "annee": [2020, 2021, 2020, 2021],
                "region": ["SAVA", "SAVA", "DIANA", "DIANA"],
                "population": [100, 110, 80, None],
                "demande_bois_chauffe_m3_ebr": [50, 55, 40, 44],
                "demande_charbon_m3_ebr": [10, 12, 8, 9],
                "demande_totale_m3_ebr": [60, 67, 48, 53],
            }
        )

    def test_statistics_include_quality_indicators(self) -> None:
        statistics = compute_statistics(self.frame)

        self.assertEqual(statistics.loc["population", "missing_count"], 1)
        self.assertEqual(statistics.loc["population", "missing_pct"], 25)
        self.assertIn("iqr_outlier_candidates", statistics.columns)

    def test_summary_describes_temporal_panel(self) -> None:
        summary = build_summary(self.frame, "demande")

        self.assertEqual(summary["observations"], 4)
        self.assertEqual(summary["nombre_regions"], 2)
        self.assertEqual(summary["annee_min"], 2020)
        self.assertEqual(summary["annee_max"], 2021)
        self.assertEqual(summary["doublons_region_annee"], 0)
        self.assertEqual(summary["valeurs_manquantes_pct"]["population"], 25)


if __name__ == "__main__":
    unittest.main()
