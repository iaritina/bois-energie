"""Reentraine les modeles finaux et predit recursivement de 2025 a 2030."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.clean_data import run_cleaning
from src.features.prepare_features import (
    BASE_FEATURES,
    TASKS,
    TEMPORAL_FEATURES,
    add_temporal_features,
)
from src.models.train_models import (
    build_model_pipeline,
    candidate_models,
    compute_metrics,
    predict_nonnegative,
)


FINAL_MODELS = {
    "demande_bois_chauffe": "gradient_boosting",
    "demande_charbon": "gradient_boosting",
    "offre_bois_feu": "random_forest",
    "offre_charbon": "random_forest",
}

FORECAST_COLUMN_NAMES = {
    "demande_bois_chauffe": "demande_bois_chauffe_m3_ebr_prevue",
    "demande_charbon": "demande_charbon_m3_ebr_prevue",
    "offre_bois_feu": "offre_bois_feu_m3_ebr_prevue",
    "offre_charbon": "offre_charbon_m3_ebr_prevue",
}


def load_historical_and_scenario(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge l'historique et les covariables du scenario futur nettoye."""
    historical_path = PROJECT_ROOT / f"data/interim/{dataset}_historique.csv"
    scenario_path = PROJECT_ROOT / f"data/interim/{dataset}_projection.csv"
    if not historical_path.exists() or not scenario_path.exists():
        run_cleaning(dataset)
    return (
        pd.read_csv(historical_path, sep=";"),
        pd.read_csv(scenario_path, sep=";"),
    )


def prepare_final_training(
    historical: pd.DataFrame, task_name: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare X et y sur toutes les observations historiques disponibles."""
    task = TASKS[task_name]
    target = task["target"]
    dataset = task["dataset"]
    prepared = add_temporal_features(historical, target)
    prepared = prepared.dropna(subset=[target])
    feature_columns = BASE_FEATURES[dataset] + TEMPORAL_FEATURES
    return prepared[feature_columns].copy(), prepared[target].copy()


def recursive_forecast(
    model: Any,
    scenario: pd.DataFrame,
    historical_targets: pd.DataFrame,
    base_features: list[str],
) -> pd.DataFrame:
    """Predit chaque annee en reutilisant uniquement les predictions anterieures."""
    target_history = {
        (str(row.region), int(row.annee)): float(row.target)
        for row in historical_targets.itertuples()
        if pd.notna(row.target)
    }
    prediction_frames: list[pd.DataFrame] = []

    for year in sorted(int(value) for value in scenario["annee"].unique()):
        year_features = scenario.loc[scenario["annee"] == year, base_features].copy()
        year_features = year_features.sort_values("region").reset_index(drop=True)

        lags = []
        rolling_means = []
        for region in year_features["region"].astype(str):
            lag = target_history.get((region, year - 1), np.nan)
            previous_values = [
                target_history.get((region, previous_year), np.nan)
                for previous_year in range(year - 3, year)
            ]
            available_values = [value for value in previous_values if np.isfinite(value)]
            lags.append(lag)
            rolling_means.append(
                float(np.mean(available_values)) if available_values else np.nan
            )

        year_features["target_lag_1"] = lags
        year_features["target_rolling_mean_3"] = rolling_means
        predictions = predict_nonnegative(model, year_features)

        year_result = year_features[["annee", "region"]].copy()
        year_result["prediction"] = predictions
        prediction_frames.append(year_result)
        for region, prediction in zip(year_result["region"], predictions):
            target_history[(str(region), year)] = float(prediction)

    return pd.concat(prediction_frames, ignore_index=True)


def train_and_forecast_task(task_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reentraine le modele recommande et genere six annees de prevision."""
    task = TASKS[task_name]
    historical, scenario = load_historical_and_scenario(task["dataset"])
    x_historical, y_historical = prepare_final_training(historical, task_name)

    model_name = FINAL_MODELS[task_name]
    estimator = candidate_models()[model_name]
    pipeline = build_model_pipeline(x_historical, estimator)
    pipeline.fit(x_historical, y_historical)

    model_dir = PROJECT_ROOT / f"models/{task_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "final_model.joblib")

    historical_targets = historical[["annee", "region", task["target"]]].rename(
        columns={task["target"]: "target"}
    )
    predictions = recursive_forecast(
        pipeline,
        scenario,
        historical_targets,
        BASE_FEATURES[task["dataset"]],
    )
    predictions = predictions.rename(
        columns={"prediction": FORECAST_COLUMN_NAMES[task_name]}
    )

    scenario_comparison = scenario[["annee", "region", task["target"]]].merge(
        predictions, on=["annee", "region"], how="left"
    )
    valid = scenario_comparison[task["target"]].notna()
    comparison_metrics = compute_metrics(
        scenario_comparison.loc[valid, task["target"]],
        scenario_comparison.loc[valid, FORECAST_COLUMN_NAMES[task_name]].to_numpy(),
    )
    scenario_comparison = scenario_comparison.rename(
        columns={task["target"]: "scenario_synthetique"}
    )
    scenario_comparison["ecart_prediction_scenario"] = (
        scenario_comparison[FORECAST_COLUMN_NAMES[task_name]]
        - scenario_comparison["scenario_synthetique"]
    )

    output_dir = PROJECT_ROOT / "data/processed/forecasts"
    metrics_dir = PROJECT_ROOT / "reports/metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output_dir / f"{task_name}_2025_2030.csv", sep=";", index=False
    )
    scenario_comparison.to_csv(
        metrics_dir / f"comparaison_scenario_{task_name}.csv", sep=";", index=False
    )

    return predictions, {
        "modele": model_name,
        "cible": task["target"],
        "observations_entrainement": int(len(x_historical)),
        "previsions": int(len(predictions)),
        "comparaison_scenario_synthetique": comparison_metrics,
    }


def assemble_energy_balance(forecasts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fusionne les quatre previsions et calcule deficit ou surplus."""
    balance: pd.DataFrame | None = None
    for task_name in TASKS:
        task_forecast = forecasts[task_name]
        balance = (
            task_forecast.copy()
            if balance is None
            else balance.merge(task_forecast, on=["annee", "region"], how="inner")
        )
    if balance is None:
        raise ValueError("Aucune prevision a fusionner")

    balance["demande_totale_m3_ebr_prevue"] = (
        balance["demande_bois_chauffe_m3_ebr_prevue"]
        + balance["demande_charbon_m3_ebr_prevue"]
    )
    balance["offre_totale_m3_ebr_prevue"] = (
        balance["offre_bois_feu_m3_ebr_prevue"]
        + balance["offre_charbon_m3_ebr_prevue"]
    )
    balance["ecart_offre_demande_m3_ebr"] = (
        balance["offre_totale_m3_ebr_prevue"]
        - balance["demande_totale_m3_ebr_prevue"]
    )
    balance["taux_couverture_pct"] = (
        balance["offre_totale_m3_ebr_prevue"]
        / balance["demande_totale_m3_ebr_prevue"]
        * 100
    )
    balance["statut"] = np.select(
        [
            balance["ecart_offre_demande_m3_ebr"] < 0,
            balance["ecart_offre_demande_m3_ebr"] > 0,
        ],
        ["deficit", "surplus"],
        default="equilibre",
    )
    return balance.sort_values(["annee", "region"]).reset_index(drop=True)


def _plot_forecast_balance(balance: pd.DataFrame, figure_dir: Path) -> None:
    annual = balance.groupby("annee", as_index=False)[
        [
            "demande_totale_m3_ebr_prevue",
            "offre_totale_m3_ebr_prevue",
            "ecart_offre_demande_m3_ebr",
        ]
    ].sum()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(
        annual["annee"],
        annual["demande_totale_m3_ebr_prevue"],
        marker="o",
        color="#9B5D2E",
        label="Demande prévue",
    )
    axes[0].plot(
        annual["annee"],
        annual["offre_totale_m3_ebr_prevue"],
        marker="o",
        color="#315C4C",
        label="Offre prévue",
    )
    axes[0].set_title("Prévisions nationales expérimentales")
    axes[0].set_xlabel("Année")
    axes[0].set_ylabel("Volume (m³ EBR)")
    axes[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    gap = balance.pivot(
        index="region", columns="annee", values="ecart_offre_demande_m3_ebr"
    )
    sns.heatmap(
        gap,
        cmap="RdYlGn",
        center=0,
        ax=axes[1],
        cbar_kws={"label": "Offre - demande (m³ EBR)"},
    )
    axes[1].set_title("Écart prévu par région")
    axes[1].set_xlabel("Année")
    axes[1].set_ylabel("Région")
    figure.tight_layout()
    figure.savefig(
        figure_dir / "bilan_offre_demande_2025_2030.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def run_forecasts() -> dict[str, Any]:
    """Produit les quatre previsions et le bilan regional 2025-2030."""
    forecasts: dict[str, pd.DataFrame] = {}
    task_reports: dict[str, Any] = {}
    for task_name in TASKS:
        predictions, task_report = train_and_forecast_task(task_name)
        forecasts[task_name] = predictions
        task_reports[task_name] = task_report

    balance = assemble_energy_balance(forecasts)
    output_dir = PROJECT_ROOT / "data/processed/forecasts"
    generated_dir = PROJECT_ROOT / "reports/generated"
    figure_dir = PROJECT_ROOT / "reports/figures/forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    balance.to_csv(
        output_dir / "bilan_offre_demande_2025_2030.csv", sep=";", index=False
    )

    annual = balance.groupby("annee", as_index=False).agg(
        demande_totale_m3_ebr_prevue=("demande_totale_m3_ebr_prevue", "sum"),
        offre_totale_m3_ebr_prevue=("offre_totale_m3_ebr_prevue", "sum"),
        ecart_offre_demande_m3_ebr=("ecart_offre_demande_m3_ebr", "sum"),
        regions_en_deficit=("statut", lambda values: int((values == "deficit").sum())),
        regions_en_surplus=("statut", lambda values: int((values == "surplus").sum())),
    )
    annual.to_csv(
        generated_dir / "previsions_nationales_2025_2030.csv", sep=";", index=False
    )
    _plot_forecast_balance(balance, figure_dir)

    report = {
        "nature": "previsions experimentales sur donnees et covariables synthetiques",
        "periode": [2025, 2030],
        "methode": "prevision recursive; les predictions alimentent les retards futurs",
        "utilisation_cibles_scenario": "comparaison uniquement, jamais comme entree des modeles",
        "taches": task_reports,
        "bilan_national": annual.to_dict(orient="records"),
    }
    with (generated_dir / "previsions_2025_2030.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = run_forecasts()
    for year in report["bilan_national"]:
        print(
            f"{year['annee']}: offre={year['offre_totale_m3_ebr_prevue']:.0f}, "
            f"demande={year['demande_totale_m3_ebr_prevue']:.0f}, "
            f"ecart={year['ecart_offre_demande_m3_ebr']:.0f}, "
            f"deficits={year['regions_en_deficit']}/22"
        )


if __name__ == "__main__":
    main()
