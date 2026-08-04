"""Tests du nettoyage des donnees offre-demande."""

import unittest

import pandas as pd

from src.data.clean_data import NUMERIC_COLUMNS, clean_dataset


class CleanDataTests(unittest.TestCase):
    def _minimal_demande(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id_observation": ["DEM-2022-06", "DEM-2022-06"],
                "annee": ["2022/2023", "2022/2023"],
                "region": ["Amoron i Mania", "Amoron i Mania"],
                "milieu_dominant": [" Rural ", " Rural "],
                "population": ["1 000", "1 000"],
                "taux_urbanisation_pct": ["20,5", "20,5"],
                "nombre_menages": ["200", "200"],
                "taille_menage_moy": ["5", "5"],
                "part_menages_bois_chauffe_pct": ["70", "70"],
                "part_menages_charbon_pct": ["25", "25"],
                "part_menages_petrole_pct": ["2", "2"],
                "part_menages_gaz_pct": ["1", "1"],
                "part_menages_electricite_pct": ["2", "2"],
                "taux_foyers_ameliores_pct": ["15", "15"],
                "cons_bois_chauffe_m3_par_hab": ["0,5", "0,5"],
                "cons_charbon_m3_ebr_par_hab": ["0.2", "0.2"],
                "prix_bois_ariary_stere": ["5 000", "5 000"],
                "prix_charbon_ariary_sac": ["2 000", "2 000"],
                "demande_bois_chauffe_m3_ebr": ["500", "500"],
                "demande_charbon_m3_ebr": ["200", "200"],
                "demande_totale_m3_ebr": ["999", "999"],
                "taux_pauvrete_estime_pct": ["101", "101"],
                "source_principale": [" Source locale ", " Source locale "],
                "statut_donnee": ["historique_synthetique"] * 2,
                "commentaire_collecte": ["ND", "ND"],
            }
        )

    def test_harmonizes_and_deduplicates_demande(self) -> None:
        cleaned, report = clean_dataset(self._minimal_demande(), "demande")

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.loc[0, "annee"], 2022)
        self.assertEqual(cleaned.loc[0, "region"], "Amoron'i Mania")
        self.assertEqual(cleaned.loc[0, "population"], 1000)
        self.assertEqual(cleaned.loc[0, "taux_urbanisation_pct"], 20.5)
        self.assertEqual(cleaned.loc[0, "demande_totale_m3_ebr"], 700)
        self.assertTrue(pd.isna(cleaned.loc[0, "taux_pauvrete_estime_pct"]))
        self.assertTrue(pd.isna(cleaned.loc[0, "commentaire_collecte"]))
        self.assertEqual(report["doublons_exacts_supprimes"], 1)
        self.assertEqual(report["totaux_recalcules"], 2)

    def test_rejects_unknown_dataset(self) -> None:
        with self.assertRaises(ValueError):
            clean_dataset(pd.DataFrame(), "inconnu")

    def test_cleans_offer_values_and_total(self) -> None:
        row = {column: "1" for column in NUMERIC_COLUMNS["offre"]}
        row.update(
            {
                "id_observation": "OFF-2020-22",
                "annee": "2020 ",
                "region": "sava",
                "source_principale": " Source locale ",
                "statut_donnee": "historique_synthetique",
                "commentaire_collecte": "-",
                "production_bois_feu_m3_ebr": "200",
                "production_bois_charbon_m3_ebr": "50",
                "offre_totale_m3_ebr": "999",
                "nombre_centres_carbonisation": "-1",
            }
        )

        cleaned, report = clean_dataset(pd.DataFrame([row]), "offre")

        self.assertEqual(cleaned.loc[0, "region"], "SAVA")
        self.assertEqual(cleaned.loc[0, "offre_totale_m3_ebr"], 250)
        self.assertTrue(pd.isna(cleaned.loc[0, "nombre_centres_carbonisation"]))
        self.assertEqual(report["regions_harmonisees"], 1)
        self.assertEqual(report["totaux_recalcules"], 1)


if __name__ == "__main__":
    unittest.main()
