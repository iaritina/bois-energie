from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEXT_COLUMNS = {
    "demande": [
        "id_observation",
        "region",
        "milieu_dominant",
        "source_principale",
        "statut_donnee",
        "commentaire_collecte",
    ],
    "offre": [
        "id_observation",
        "region",
        "source_principale",
        "statut_donnee",
        "commentaire_collecte",
    ],
}

NUMERIC_COLUMNS = {
    "demande": [
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
        "cons_bois_chauffe_m3_par_hab",
        "cons_charbon_m3_ebr_par_hab",
        "prix_bois_ariary_stere",
        "prix_charbon_ariary_sac",
        "demande_bois_chauffe_m3_ebr",
        "demande_charbon_m3_ebr",
        "demande_totale_m3_ebr",
        "taux_pauvrete_estime_pct",
    ],
    "offre": [
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
        "production_bois_feu_m3_ebr",
        "production_bois_charbon_m3_ebr",
        "production_charbon_tonnes",
        "offre_totale_m3_ebr",
        "volume_transporte_tonnes",
        "distance_moyenne_marche_km",
        "cout_transport_ariary_tonne_km",
        "prix_producteur_charbon_ariary_sac",
    ],
}

PERCENTAGE_COLUMNS = {
    "demande": [
        "taux_urbanisation_pct",
        "part_menages_bois_chauffe_pct",
        "part_menages_charbon_pct",
        "part_menages_petrole_pct",
        "part_menages_gaz_pct",
        "part_menages_electricite_pct",
        "taux_foyers_ameliores_pct",
        "taux_pauvrete_estime_pct",
    ],
    "offre": [
        "taux_deforestation_pct",
        "rendement_carbonisation_pct",
        "part_carbonisation_amelioree_pct",
    ],
}

TOTAL_COMPONENTS = {
    "demande": (
        "demande_totale_m3_ebr",
        "demande_bois_chauffe_m3_ebr",
        "demande_charbon_m3_ebr",
    ),
    "offre": (
        "offre_totale_m3_ebr",
        "production_bois_feu_m3_ebr",
        "production_bois_charbon_m3_ebr",
    ),
}

RAW_FILES = {
    "demande": PROJECT_ROOT
    / "data/raw/demande/demande_bois_energie_brute_synthetique.csv",
    "offre": PROJECT_ROOT / "data/raw/offre/offre_bois_energie_brute_synthetique.csv",
}

CANONICAL_REGIONS = [
    "Analamanga",
    "Vakinankaratra",
    "Itasy",
    "Bongolava",
    "Haute Matsiatra",
    "Amoron'i Mania",
    "Vatovavy-Fitovinany",
    "Ihorombe",
    "Atsimo-Atsinanana",
    "Atsinanana",
    "Analanjirofo",
    "Alaotra-Mangoro",
    "Boeny",
    "Sofia",
    "Betsiboka",
    "Melaky",
    "Atsimo-Andrefana",
    "Androy",
    "Anosy",
    "Menabe",
    "DIANA",
    "SAVA",
]

MISSING_MARKERS = {"", "-", "--", "nd", "n/d", "na", "n/a", "null", "none"}


def _region_key(value: str) -> str:
    value = value.strip().casefold().replace("'", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value)


REGION_LOOKUP = {_region_key(region): region for region in CANONICAL_REGIONS}


def _clean_text_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    cleaned = re.sub(r"\s+", " ", str(value).strip())
    return pd.NA if cleaned.casefold() in MISSING_MARKERS else cleaned


def _normalize_category(value: object) -> object:
    cleaned = _clean_text_value(value)
    if pd.isna(cleaned):
        return pd.NA
    normalized = str(cleaned).casefold()
    normalized = re.sub(r"\s*\+\s*", "+", normalized)
    return re.sub(r"\s+", "_", normalized)


def _parse_numeric(series: pd.Series) -> tuple[pd.Series, int]:
    text = series.map(_clean_text_value).astype("string")
    compact = text.str.replace(r"[\s\u00a0\u202f]", "", regex=True)
    compact = compact.str.replace(",", ".", regex=False)
    parsed = pd.to_numeric(compact, errors="coerce")
    parse_errors = int((text.notna() & parsed.isna()).sum())
    return parsed, parse_errors


def _repair_years(frame: pd.DataFrame) -> tuple[pd.Series, int]:
    raw_year, _ = _parse_numeric(frame["annee"])
    id_year = pd.to_numeric(
        frame["id_observation"]
        .astype("string")
        .str.extract(r"-(\d{4})-", expand=False),
        errors="coerce",
    )
    repaired = id_year.fillna(raw_year).astype("Int64")
    raw_normalized = raw_year.astype("Int64")
    changed = int((raw_normalized.fillna(-1) != repaired.fillna(-1)).sum())
    return repaired, changed


def _harmonize_regions(series: pd.Series) -> tuple[pd.Series, int, list[str]]:
    cleaned = series.map(_clean_text_value)
    harmonized = cleaned.map(
        lambda value: (
            REGION_LOOKUP.get(_region_key(str(value)), value)
            if pd.notna(value)
            else pd.NA
        )
    )
    changed = int(
        sum(
            pd.notna(before) and pd.notna(after) and str(before) != str(after)
            for before, after in zip(cleaned, harmonized)
        )
    )
    unknown = sorted(
        {str(value) for value in harmonized.dropna() if value not in CANONICAL_REGIONS}
    )
    return harmonized.astype("string"), changed, unknown


def _missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int(frame[column].isna().sum()) for column in frame.columns}


def clean_dataset(raw: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Nettoie un DataFrame d'offre ou de demande et retourne son rapport qualite."""
    if kind not in NUMERIC_COLUMNS:
        raise ValueError("kind doit etre 'demande' ou 'offre'")

    required = set(TEXT_COLUMNS[kind] + NUMERIC_COLUMNS[kind] + ["annee"])
    missing_columns = sorted(required.difference(raw.columns))
    if missing_columns:
        raise ValueError(f"Colonnes manquantes pour {kind}: {missing_columns}")

    frame = raw.copy()
    frame.columns = [column.strip().casefold() for column in frame.columns]
    initial_rows = len(frame)

    for column in TEXT_COLUMNS[kind]:
        frame[column] = frame[column].map(_clean_text_value).astype("string")

    frame["id_observation"] = frame["id_observation"].str.upper()
    frame["annee"], repaired_years = _repair_years(frame)
    frame["region"], harmonized_regions, unknown_regions = _harmonize_regions(
        frame["region"]
    )

    for column in (
        "milieu_dominant",
        "source_principale",
        "statut_donnee",
        "commentaire_collecte",
    ):
        if column in frame.columns:
            frame[column] = frame[column].map(_normalize_category).astype("string")

    parse_errors: dict[str, int] = {}
    for column in NUMERIC_COLUMNS[kind]:
        frame[column], errors = _parse_numeric(frame[column])
        if errors:
            parse_errors[column] = errors

    invalid_negative: dict[str, int] = {}
    for column in NUMERIC_COLUMNS[kind]:
        mask = frame[column] < 0
        count = int(mask.fillna(False).sum())
        if count:
            invalid_negative[column] = count
            frame.loc[mask, column] = pd.NA

    invalid_percentages: dict[str, int] = {}
    for column in PERCENTAGE_COLUMNS[kind]:
        mask = (frame[column] < 0) | (frame[column] > 100)
        count = int(mask.fillna(False).sum())
        if count:
            invalid_percentages[column] = count
            frame.loc[mask, column] = pd.NA

    total_column, first_component, second_component = TOTAL_COMPONENTS[kind]
    calculated_total = frame[first_component] + frame[second_component]
    comparable = calculated_total.notna()
    inconsistent_total = comparable & (
        frame[total_column].isna()
        | ((frame[total_column] - calculated_total).abs() > 0.05)
    )
    corrected_totals = int(inconsistent_total.sum())
    frame.loc[inconsistent_total, total_column] = calculated_total[inconsistent_total]

    exact_duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()

    duplicate_ids = frame["id_observation"].duplicated(keep=False)
    duplicate_keys = frame.duplicated(["region", "annee"], keep=False)
    if duplicate_ids.any() or duplicate_keys.any():
        raise ValueError(
            f"{kind}: doublons conflictuels apres nettoyage "
            f"(identifiants={int(duplicate_ids.sum())}, "
            f"region-annee={int(duplicate_keys.sum())})"
        )

    frame = frame.sort_values(["annee", "region"]).reset_index(drop=True)
    frame[NUMERIC_COLUMNS[kind]] = frame[NUMERIC_COLUMNS[kind]].round(6)

    status_counts = {
        str(status): int(count)
        for status, count in frame["statut_donnee"].value_counts(dropna=False).items()
    }
    report: dict[str, Any] = {
        "jeu_donnees": kind,
        "lignes_brutes": initial_rows,
        "lignes_nettoyees": len(frame),
        "doublons_exacts_supprimes": exact_duplicates,
        "annees_reparees_depuis_identifiant": repaired_years,
        "regions_harmonisees": harmonized_regions,
        "regions_inconnues": unknown_regions,
        "erreurs_conversion_numerique": parse_errors,
        "valeurs_negatives_remplacees_par_na": invalid_negative,
        "pourcentages_hors_bornes_remplaces_par_na": invalid_percentages,
        "totaux_recalcules": corrected_totals,
        "valeurs_manquantes_finales": _missing_counts(frame),
        "repartition_statut": status_counts,
        "annee_min": int(frame["annee"].min()),
        "annee_max": int(frame["annee"].max()),
        "nombre_regions": int(frame["region"].nunique()),
    }
    return frame, report


def run_cleaning(kind: str) -> tuple[Path, dict[str, Any]]:
    """Charge, nettoie et sauvegarde un jeu de donnees."""
    raw = pd.read_csv(RAW_FILES[kind], sep=";", dtype=str, keep_default_na=False)
    cleaned, report = clean_dataset(raw, kind)

    interim_dir = PROJECT_ROOT / "data/interim"
    report_dir = PROJECT_ROOT / "reports/generated"
    interim_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_path = interim_dir / f"{kind}_nettoyee.csv"
    cleaned.to_csv(output_path, sep=";", index=False, encoding="utf-8")

    for status, suffix in (
        ("historique_synthetique", "historique"),
        ("projection_scenario", "projection"),
    ):
        subset = cleaned.loc[cleaned["statut_donnee"] == status]
        subset.to_csv(
            interim_dir / f"{kind}_{suffix}.csv",
            sep=";",
            index=False,
            encoding="utf-8",
        )

    with (report_dir / f"qualite_{kind}.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return output_path, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("demande", "offre", "tous"),
        default="tous",
        help="Jeu de donnees a nettoyer (defaut: tous).",
    )
    args = parser.parse_args()

    kinds = ("demande", "offre") if args.dataset == "tous" else (args.dataset,)
    for kind in kinds:
        output_path, report = run_cleaning(kind)
        print(
            f"{kind.capitalize()}: {report['lignes_brutes']} -> "
            f"{report['lignes_nettoyees']} lignes, sortie: {output_path}"
        )


if __name__ == "__main__":
    main()
