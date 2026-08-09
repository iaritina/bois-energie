"""Prepare les jeux chronologiques pour les modeles d'offre et de demande."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.clean_data import run_cleaning


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TASKS: dict[str, dict[str, str]] = {
    "demande_bois_chauffe": {
        "dataset": "demande",
        "target": "demande_bois_chauffe_m3_ebr",
    },
    "demande_charbon": {
        "dataset": "demande",
        "target": "demande_charbon_m3_ebr",
    },
    "offre_bois_feu": {
        "dataset": "offre",
        "target": "production_bois_feu_m3_ebr",
    },
    "offre_charbon": {
        "dataset": "offre",
        "target": "production_bois_charbon_m3_ebr",
    },
}

BASE_FEATURES = {
    "demande": [
        "annee",
        "region",
        "milieu_dominant",
        "population",
        "taux_urbanisation_pct",
        "nombre_menages",
        "taille_menage_moy",
        "part_menages_bois_chauffe_pct",
        "part_menages_charbon_pct",
        "part_menages_petrole_pct",
        "part_menages_gaz_pct",
        "part_menages_electricite_pct",
        "taux_foyers_ameliores_pct",
        "prix_bois_ariary_stere",
        "prix_charbon_ariary_sac",
        "taux_pauvrete_estime_pct",
    ],
    "offre": [
        "annee",
        "region",
        "superficie_foret_hors_ap_ha",
        "superficie_foret_humide_ha",
        "superficie_foret_seche_ha",
        "superficie_foret_epineuse_ha",
        "superficie_mangrove_ha",
        "superficie_pin_ha",
        "superficie_eucalyptus_ha",
        "reboisement_annuel_ha",
        "taux_deforestation_pct",
        "rendement_carbonisation_pct",
        "part_carbonisation_amelioree_pct",
        "nombre_centres_carbonisation",
        "distance_moyenne_marche_km",
        "cout_transport_ariary_tonne_km",
        "prix_producteur_charbon_ariary_sac",
    ],
}

EXCLUDED_FEATURES = {
    "demande": {
        "id_observation": "identifiant, sans signification predictive",
        "source_principale": "metadonnee de collecte",
        "statut_donnee": "constant dans le jeu historique",
        "commentaire_collecte": "texte de controle qualite",
        "cons_bois_chauffe_m3_par_hab": "composante directe du calcul de la demande",
        "cons_charbon_m3_ebr_par_hab": "composante directe du calcul de la demande",
        "demande_bois_chauffe_m3_ebr": "cible potentielle",
        "demande_charbon_m3_ebr": "cible potentielle",
        "demande_totale_m3_ebr": "total calcule a partir des deux cibles",
    },
    "offre": {
        "id_observation": "identifiant, sans signification predictive",
        "source_principale": "metadonnee de collecte",
        "statut_donnee": "constant dans le jeu historique",
        "commentaire_collecte": "texte de controle qualite",
        "production_bois_feu_m3_ebr": "cible potentielle",
        "production_bois_charbon_m3_ebr": "cible potentielle",
        "production_charbon_tonnes": "conversion directe de la production de charbon",
        "offre_totale_m3_ebr": "total calcule a partir des deux cibles",
        "volume_transporte_tonnes": "mesure probablement disponible apres la production",
    },
}

TEMPORAL_FEATURES = ["target_lag_1", "target_rolling_mean_3"]

SPLIT_PERIODS = {
    "train": (2000, 2018),
    "validation": (2019, 2021),
    "test": (2022, 2024),
}


def load_historical(dataset: str) -> pd.DataFrame:
    """Charge le fichier historique nettoye d'un jeu de donnees."""
    path = PROJECT_ROOT / f"data/interim/{dataset}_historique.csv"
    if not path.exists():
        run_cleaning(dataset)
    frame = pd.read_csv(path, sep=";")
    if "statut_donnee" in frame.columns:
        frame = frame.loc[frame["statut_donnee"] == "historique_synthetique"]
    return frame.copy()


def add_temporal_features(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Ajoute des variables de cible passee, sans utiliser l'annee courante."""
    result = frame.sort_values(["region", "annee"]).copy()
    grouped_target = result.groupby("region", sort=False)[target]
    result["target_lag_1"] = grouped_target.shift(1)
    result["target_rolling_mean_3"] = grouped_target.transform(
        lambda values: values.shift(1).rolling(window=3, min_periods=1).mean()
    )
    return result


def split_chronologically(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Separe le panel selon les periodes fixes du projet."""
    splits: dict[str, pd.DataFrame] = {}
    for split_name, (start_year, end_year) in SPLIT_PERIODS.items():
        mask = frame["annee"].between(start_year, end_year)
        splits[split_name] = frame.loc[mask].reset_index(drop=True)
    return splits


def prepare_task(frame: pd.DataFrame, task_name: str) -> dict[str, pd.DataFrame]:
    """Selectionne les variables et prepare les trois blocs d'une tache ML."""
    if task_name not in TASKS:
        raise ValueError(f"Tache inconnue: {task_name}")

    task = TASKS[task_name]
    dataset = task["dataset"]
    target = task["target"]
    required = set(BASE_FEATURES[dataset] + [target])
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Colonnes manquantes pour {task_name}: {missing_columns}")

    prepared = add_temporal_features(frame, target)
    selected_columns = BASE_FEATURES[dataset] + TEMPORAL_FEATURES + [target]
    prepared = prepared[selected_columns].copy()

    # Une cible ne doit jamais etre inventee par imputation.
    prepared = prepared.dropna(subset=[target])
    prepared = prepared.rename(columns={target: "target"})
    return split_chronologically(prepared)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Construit le preprocesseur a ajuster uniquement sur X_train."""
    categorical_columns = features.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = features.select_dtypes(include="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        verbose_feature_names_out=False,
    )


def _split_report(split: pd.DataFrame) -> dict[str, Any]:
    features = split.drop(columns="target")
    return {
        "lignes": int(len(split)),
        "annee_min": int(split["annee"].min()),
        "annee_max": int(split["annee"].max()),
        "valeurs_manquantes_features": int(features.isna().sum().sum()),
        "colonnes_avec_manquants": {
            column: int(count)
            for column, count in features.isna().sum().items()
            if count > 0
        },
    }


def run_preparation(dataset: str = "tous") -> dict[str, Any]:
    """Prepare et sauvegarde toutes les taches d'un ou des deux jeux."""
    if dataset not in {"demande", "offre", "tous"}:
        raise ValueError("dataset doit etre 'demande', 'offre' ou 'tous'")

    selected_tasks = {
        name: config
        for name, config in TASKS.items()
        if dataset == "tous" or config["dataset"] == dataset
    }
    loaded_frames: dict[str, pd.DataFrame] = {}
    report: dict[str, Any] = {
        "decoupage_chronologique": SPLIT_PERIODS,
        "variables_decalage_temporel": {
            "target_lag_1": "valeur de la cible de l'annee precedente",
            "target_rolling_mean_3": "moyenne des trois annees precedentes",
        },
        "pretraitement_modele": {
            "numerique": "imputation mediane puis standardisation",
            "categoriel": "imputation par modalite frequente puis encodage one-hot",
            "regle_anti_fuite": "ajuster le preprocesseur uniquement sur X_train",
        },
        "taches": {},
    }

    for task_name, task in selected_tasks.items():
        task_dataset = task["dataset"]
        if task_dataset not in loaded_frames:
            loaded_frames[task_dataset] = load_historical(task_dataset)
        splits = prepare_task(loaded_frames[task_dataset], task_name)

        output_dir = PROJECT_ROOT / f"data/processed/{task_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split in splits.items():
            split.to_csv(output_dir / f"{split_name}.csv", sep=";", index=False)

        report["taches"][task_name] = {
            "jeu_donnees": task_dataset,
            "cible_originale": task["target"],
            "variables_retenues": BASE_FEATURES[task_dataset] + TEMPORAL_FEATURES,
            "variables_exclues": EXCLUDED_FEATURES[task_dataset],
            "blocs": {
                split_name: _split_report(split)
                for split_name, split in splits.items()
            },
        }

    report_dir = PROJECT_ROOT / "reports/generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "preparation_ml.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("demande", "offre", "tous"),
        default="tous",
        help="Jeu de donnees a preparer (defaut: tous).",
    )
    args = parser.parse_args()
    report = run_preparation(args.dataset)
    for task_name, task_report in report["taches"].items():
        counts = task_report["blocs"]
        print(
            f"{task_name}: train={counts['train']['lignes']}, "
            f"validation={counts['validation']['lignes']}, "
            f"test={counts['test']['lignes']}"
        )


if __name__ == "__main__":
    main()
