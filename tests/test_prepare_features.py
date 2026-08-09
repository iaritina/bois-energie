"""Tests de selection et preparation des variables ML."""

import unittest

import numpy as np
import pandas as pd

from src.features.prepare_features import (
    BASE_FEATURES,
    add_temporal_features,
    build_preprocessor,
    prepare_task,
    split_chronologically,
)


class PrepareFeaturesTests(unittest.TestCase):
    def test_temporal_features_use_only_previous_years(self) -> None:
        frame = pd.DataFrame(
            {
                "region": ["SAVA", "SAVA", "SAVA", "DIANA", "DIANA"],
                "annee": [2022, 2020, 2021, 2020, 2021],
                "cible": [30.0, 10.0, 20.0, 5.0, 8.0],
            }
        )

        result = add_temporal_features(frame, "cible")
        sava_2022 = result.loc[
            (result["region"] == "SAVA") & (result["annee"] == 2022)
        ].iloc[0]

        self.assertEqual(sava_2022["target_lag_1"], 20.0)
        self.assertEqual(sava_2022["target_rolling_mean_3"], 15.0)

    def test_chronological_split_respects_boundaries(self) -> None:
        frame = pd.DataFrame(
            {
                "annee": [2000, 2018, 2019, 2021, 2022, 2024],
                "target": range(6),
            }
        )

        splits = split_chronologically(frame)

        self.assertEqual(splits["train"]["annee"].tolist(), [2000, 2018])
        self.assertEqual(splits["validation"]["annee"].tolist(), [2019, 2021])
        self.assertEqual(splits["test"]["annee"].tolist(), [2022, 2024])

    def test_prepare_task_keeps_only_approved_features(self) -> None:
        rows = 4
        frame = pd.DataFrame(
            {column: [1.0] * rows for column in BASE_FEATURES["demande"]}
        )
        frame["annee"] = [2017, 2018, 2019, 2022]
        frame["region"] = ["SAVA"] * rows
        frame["milieu_dominant"] = ["rural"] * rows
        frame["demande_bois_chauffe_m3_ebr"] = [10.0, 12.0, 13.0, 15.0]
        frame["demande_totale_m3_ebr"] = [99.0] * rows
        frame["cons_bois_chauffe_m3_par_hab"] = [99.0] * rows

        splits = prepare_task(frame, "demande_bois_chauffe")
        all_columns = set(splits["train"].columns)

        self.assertIn("target_lag_1", all_columns)
        self.assertIn("target", all_columns)
        self.assertNotIn("demande_totale_m3_ebr", all_columns)
        self.assertNotIn("cons_bois_chauffe_m3_par_hab", all_columns)

    def test_preprocessor_imputes_and_encodes(self) -> None:
        features = pd.DataFrame(
            {
                "population": [100.0, np.nan, 150.0],
                "region": ["SAVA", "DIANA", "SAVA"],
            }
        )
        preprocessor = build_preprocessor(features)

        transformed = preprocessor.fit_transform(features)

        self.assertFalse(np.isnan(transformed).any())
        self.assertGreater(transformed.shape[1], features.shape[1])


if __name__ == "__main__":
    unittest.main()
