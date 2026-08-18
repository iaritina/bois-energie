"""Tests des indicateurs et niveaux de vulnerabilite regionaux."""

import unittest

import pandas as pd

from src.evaluation.regional_indicators import (
    classify_coverage,
    compute_regional_indicators,
)


class RegionalIndicatorsTests(unittest.TestCase):
    def test_coverage_classification_boundaries(self) -> None:
        self.assertEqual(classify_coverage(49.9), "critique")
        self.assertEqual(classify_coverage(50.0), "elevee")
        self.assertEqual(classify_coverage(75.0), "moderee")
        self.assertEqual(classify_coverage(95.0), "equilibre")
        self.assertEqual(classify_coverage(105.0), "equilibre")
        self.assertEqual(classify_coverage(105.1), "surplus")

    def test_regional_aggregation_and_ranking(self) -> None:
        balance = pd.DataFrame(
            {
                "annee": [2025, 2026, 2025, 2026],
                "region": ["SAVA", "SAVA", "DIANA", "DIANA"],
                "demande_totale_m3_ebr_prevue": [100.0, 100.0, 100.0, 100.0],
                "offre_totale_m3_ebr_prevue": [40.0, 60.0, 100.0, 100.0],
                "ecart_offre_demande_m3_ebr": [-60.0, -40.0, 0.0, 0.0],
                "taux_couverture_pct": [40.0, 60.0, 100.0, 100.0],
                "statut": ["deficit", "deficit", "equilibre", "equilibre"],
            }
        )

        indicators = compute_regional_indicators(balance)
        sava = indicators.loc[indicators["region"] == "SAVA"].iloc[0]
        diana = indicators.loc[indicators["region"] == "DIANA"].iloc[0]

        self.assertEqual(sava["rang_vulnerabilite"], 1)
        self.assertEqual(sava["taux_couverture_global_pct"], 50)
        self.assertEqual(sava["deficit_cumule_m3_ebr"], 100)
        self.assertEqual(sava["annees_en_deficit"], 2)
        self.assertEqual(sava["niveau_vulnerabilite"], "elevee")
        self.assertEqual(diana["niveau_vulnerabilite"], "equilibre")


if __name__ == "__main__":
    unittest.main()
