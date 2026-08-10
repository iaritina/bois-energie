"""Tests des previsions recursives et du bilan offre-demande."""

import unittest

import numpy as np
import pandas as pd

from src.forecasting.forecast_2030 import (
    FORECAST_COLUMN_NAMES,
    assemble_energy_balance,
    recursive_forecast,
)


class LagPlusOneModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["target_lag_1"].fillna(0).to_numpy() + 1


class ForecastTests(unittest.TestCase):
    def test_recursive_forecast_reuses_previous_prediction(self) -> None:
        scenario = pd.DataFrame(
            {
                "annee": [2025, 2026],
                "region": ["SAVA", "SAVA"],
                "population": [100.0, 101.0],
                "future_target_that_must_not_be_used": [999.0, 999.0],
            }
        )
        history = pd.DataFrame(
            {"annee": [2024], "region": ["SAVA"], "target": [10.0]}
        )

        forecast = recursive_forecast(
            LagPlusOneModel(),
            scenario,
            history,
            ["annee", "region", "population"],
        )

        self.assertEqual(forecast["prediction"].tolist(), [11.0, 12.0])

    def test_balance_calculates_totals_and_status(self) -> None:
        values = {
            "demande_bois_chauffe": 60.0,
            "demande_charbon": 40.0,
            "offre_bois_feu": 50.0,
            "offre_charbon": 20.0,
        }
        forecasts = {
            task: pd.DataFrame(
                {
                    "annee": [2025],
                    "region": ["SAVA"],
                    FORECAST_COLUMN_NAMES[task]: [value],
                }
            )
            for task, value in values.items()
        }

        balance = assemble_energy_balance(forecasts)

        self.assertEqual(balance.loc[0, "demande_totale_m3_ebr_prevue"], 100)
        self.assertEqual(balance.loc[0, "offre_totale_m3_ebr_prevue"], 70)
        self.assertEqual(balance.loc[0, "ecart_offre_demande_m3_ebr"], -30)
        self.assertEqual(balance.loc[0, "taux_couverture_pct"], 70)
        self.assertEqual(balance.loc[0, "statut"], "deficit")


if __name__ == "__main__":
    unittest.main()
