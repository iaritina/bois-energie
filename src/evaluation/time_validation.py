"""Valide les modeles sur plusieurs fenetres temporelles croissantes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.features.prepare_features import TASKS
from src.models.train_models import (
    build_model_pipeline,
    candidate_models,
    compute_metrics,
    load_task_splits,
    predict_nonnegative,
)


TIME_FOLDS = (
    {
        "name": "fenetre_1",
        "train_start": 2000,
        "train_end": 2012,
        "validation_start": 2013,
        "validation_end": 2015,
    },
    {
        "name": "fenetre_2",
        "train_start": 2000,
        "train_end": 2015,
        "validation_start": 2016,
        "validation_end": 2018,
    },
    {
        "name": "fenetre_3",
        "train_start": 2000,
        "train_end": 2018,
        "validation_start": 2019,
        "validation_end": 2021,
    },
)


def create_fold_frames(
    history: pd.DataFrame,
) -> list[tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]]:
    """Construit les blocs d'entrainement et validation de chaque fenetre."""
    folds = []
    for definition in TIME_FOLDS:
        train_mask = history["annee"].between(
            definition["train_start"], definition["train_end"]
        )
        validation_mask = history["annee"].between(
            definition["validation_start"], definition["validation_end"]
        )
        train = history.loc[train_mask].reset_index(drop=True)
        validation = history.loc[validation_mask].reset_index(drop=True)
        if train.empty or validation.empty:
            raise ValueError(f"Fenetre vide: {definition['name']}")
        folds.append((definition, train, validation))
    return folds


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Resume la performance moyenne et sa variabilite pour chaque modele."""
    return (
        results.groupby(["tache", "modele"], as_index=False)
        .agg(
            wape_moyen_pct=("wape_pct", "mean"),
            wape_ecart_type_pct=("wape_pct", "std"),
            wape_min_pct=("wape_pct", "min"),
            wape_max_pct=("wape_pct", "max"),
            mae_moyenne=("mae", "mean"),
            rmse_moyenne=("rmse", "mean"),
            r2_moyen=("r2", "mean"),
        )
        .sort_values(["tache", "wape_moyen_pct", "wape_ecart_type_pct"])
        .reset_index(drop=True)
    )


def recommend_model(task_summary: pd.DataFrame) -> str:
    """Retient le WAPE moyen minimal, puis sa variabilite et la MAE."""
    if task_summary.empty:
        raise ValueError("Resume temporel vide")
    ordered = task_summary.sort_values(
        ["wape_moyen_pct", "wape_ecart_type_pct", "mae_moyenne"]
    )
    return str(ordered.iloc[0]["modele"])


def _load_pretest_history(task_name: str) -> pd.DataFrame:
    splits = load_task_splits(task_name)
    history = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    if int(history["annee"].max()) > 2021:
        raise ValueError("La validation temporelle ne doit pas utiliser le test 2022-2024")
    return history.sort_values(["annee", "region"]).reset_index(drop=True)


def validate_task(task_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalue tous les modeles sur les trois fenetres d'une tache."""
    history = _load_pretest_history(task_name)
    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for definition, train, validation in create_fold_frames(history):
        x_train = train.drop(columns="target")
        y_train = train["target"]
        x_validation = validation.drop(columns="target")
        y_validation = validation["target"]

        for model_name, estimator in candidate_models().items():
            pipeline = build_model_pipeline(x_train, estimator)
            pipeline.fit(x_train, y_train)
            predicted = predict_nonnegative(pipeline, x_validation)
            metrics = compute_metrics(y_validation, predicted)
            result_rows.append(
                {
                    "tache": task_name,
                    "fenetre": definition["name"],
                    "entrainement": f"{definition['train_start']}-{definition['train_end']}",
                    "validation": f"{definition['validation_start']}-{definition['validation_end']}",
                    "modele": model_name,
                    "lignes_entrainement": len(train),
                    "lignes_validation": len(validation),
                    **metrics,
                }
            )
            prediction_frame = pd.DataFrame(
                {
                    "tache": task_name,
                    "fenetre": definition["name"],
                    "modele": model_name,
                    "annee": x_validation["annee"].to_numpy(),
                    "region": x_validation["region"].to_numpy(),
                    "observe": y_validation.to_numpy(),
                    "predit": predicted,
                }
            )
            prediction_frame["erreur"] = (
                prediction_frame["predit"] - prediction_frame["observe"]
            )
            prediction_frame["erreur_absolue"] = prediction_frame["erreur"].abs()
            prediction_rows.append(prediction_frame)

    return pd.DataFrame(result_rows), pd.concat(prediction_rows, ignore_index=True)


def _regional_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        predictions.groupby(["tache", "modele", "region"], as_index=False)
        .agg(
            observations=("observe", "size"),
            mae=("erreur_absolue", "mean"),
            erreur_absolue_totale=("erreur_absolue", "sum"),
            volume_observe_total=("observe", lambda values: values.abs().sum()),
        )
    )
    grouped["wape_pct"] = (
        grouped["erreur_absolue_totale"] / grouped["volume_observe_total"] * 100
    )
    return grouped.sort_values(["tache", "modele", "wape_pct"], ascending=[True, True, False])


def _plot_wape(results: pd.DataFrame, recommendations: dict[str, str], path: Path) -> None:
    task_names = list(recommendations)
    columns = min(2, len(task_names))
    rows = math.ceil(len(task_names) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(13, 4.5 * rows),
        sharex=True,
        squeeze=False,
    )
    colors = {
        "ridge": "#9B5D2E",
        "random_forest": "#315C4C",
        "gradient_boosting": "#D69E2E",
    }
    for axis, task_name in zip(axes.flat, task_names):
        task_results = results.loc[results["tache"] == task_name]
        for model_name, model_results in task_results.groupby("modele"):
            model_results = model_results.sort_values("fenetre")
            label = model_name.replace("_", " ")
            if recommendations[task_name] == model_name:
                label += " (recommandé)"
            axis.plot(
                model_results["fenetre"],
                model_results["wape_pct"],
                marker="o",
                linewidth=2,
                color=colors[model_name],
                label=label,
            )
        axis.set_title(task_name.replace("_", " ").title())
        axis.set_ylabel("WAPE (%)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    for unused_axis in list(axes.flat)[len(task_names):]:
        unused_axis.set_axis_off()
    figure.suptitle("Stabilité temporelle des modèles", fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_time_validation(dataset: str = "tous") -> dict[str, Any]:
    """Execute la validation temporelle pour la demande, l'offre ou les deux."""
    if dataset not in {"demande", "offre", "tous"}:
        raise ValueError("dataset doit etre 'demande', 'offre' ou 'tous'")
    selected_tasks = [
        task_name
        for task_name, config in TASKS.items()
        if dataset == "tous" or config["dataset"] == dataset
    ]

    all_results = []
    all_predictions = []
    for task_name in selected_tasks:
        results, predictions = validate_task(task_name)
        all_results.append(results)
        all_predictions.append(predictions)

    results = pd.concat(all_results, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_results(results)
    recommendations = {
        task_name: recommend_model(summary.loc[summary["tache"] == task_name])
        for task_name in selected_tasks
    }
    regional_errors = _regional_error_summary(predictions)

    metrics_dir = PROJECT_ROOT / "reports/metrics"
    figure_dir = PROJECT_ROOT / "reports/figures/models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(metrics_dir / "validation_temporelle.csv", sep=";", index=False)
    summary.to_csv(
        metrics_dir / "validation_temporelle_resume.csv", sep=";", index=False
    )
    predictions.to_csv(
        metrics_dir / "validation_temporelle_predictions.csv", sep=";", index=False
    )
    regional_errors.to_csv(
        metrics_dir / "validation_temporelle_regions.csv", sep=";", index=False
    )
    _plot_wape(
        results,
        recommendations,
        figure_dir / "validation_temporelle_wape.png",
    )

    report = {
        "protocole": {
            "fenetres": list(TIME_FOLDS),
            "annees_test_exclues": [2022, 2023, 2024],
            "classement": "WAPE moyen, puis ecart-type du WAPE, puis MAE moyenne",
        },
        "recommandations": recommendations,
        "resume": {
            task_name: summary.loc[summary["tache"] == task_name]
            .to_dict(orient="records")
            for task_name in selected_tasks
        },
    }
    with (metrics_dir / "validation_temporelle.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("demande", "offre", "tous"),
        default="tous",
        help="Jeux a valider (defaut: tous).",
    )
    args = parser.parse_args()
    report = run_time_validation(args.dataset)
    for task_name, model_name in report["recommandations"].items():
        task_summary = report["resume"][task_name]
        selected = next(row for row in task_summary if row["modele"] == model_name)
        print(
            f"{task_name}: recommande={model_name}, "
            f"WAPE moyen={selected['wape_moyen_pct']:.2f} %, "
            f"ecart-type={selected['wape_ecart_type_pct']:.2f}"
        )


if __name__ == "__main__":
    main()
