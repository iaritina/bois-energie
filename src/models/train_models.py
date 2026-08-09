"""Entraine et compare les modeles classiques pour les quatre cibles."""

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
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.features.prepare_features import TASKS, build_preprocessor, run_preparation


RANDOM_STATE = 42


def candidate_models() -> dict[str, Any]:
    """Retourne une nouvelle instance de chaque modele candidat."""
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=3,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
    }


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule les quatre metriques dans l'unite originale de la cible."""
    true_values = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    absolute_total = np.abs(true_values).sum()
    wape = np.abs(true_values - predictions).sum() / absolute_total * 100
    return {
        "mae": float(mean_absolute_error(true_values, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(true_values, predictions))),
        "r2": float(r2_score(true_values, predictions)),
        "wape_pct": float(wape),
    }


def select_best_model(validation_metrics: dict[str, dict[str, float]]) -> str:
    """Selectionne le plus faible WAPE, puis la plus faible MAE en cas d'egalite."""
    if not validation_metrics:
        raise ValueError("Aucune metrique de validation fournie")
    return min(
        validation_metrics,
        key=lambda name: (
            validation_metrics[name]["wape_pct"],
            validation_metrics[name]["mae"],
        ),
    )


def build_model_pipeline(features: pd.DataFrame, estimator: Any) -> Pipeline:
    """Associe pretraitement et regression sur log(1 + cible)."""
    transformed_estimator = TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=True,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("regressor", transformed_estimator),
        ]
    )


def predict_nonnegative(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Ramene a zero une eventuelle prediction negative non physique."""
    return np.clip(model.predict(features), a_min=0, a_max=None)


def load_task_splits(task_name: str) -> dict[str, pd.DataFrame]:
    """Charge les trois blocs prepares et les regenere si necessaire."""
    if task_name not in TASKS:
        raise ValueError(f"Tache inconnue: {task_name}")
    task_dir = PROJECT_ROOT / f"data/processed/{task_name}"
    paths = {name: task_dir / f"{name}.csv" for name in ("train", "validation", "test")}
    if not all(path.exists() for path in paths.values()):
        run_preparation(TASKS[task_name]["dataset"])
    return {name: pd.read_csv(path, sep=";") for name, path in paths.items()}


def _split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns="target"), frame["target"]


def _extract_importance(model: Pipeline) -> pd.DataFrame:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    transformed = model.named_steps["regressor"].regressor_
    if hasattr(transformed, "feature_importances_"):
        values = transformed.feature_importances_
    elif hasattr(transformed, "coef_"):
        values = np.abs(np.ravel(transformed.coef_))
    else:
        return pd.DataFrame(columns=["variable", "importance"])
    return (
        pd.DataFrame({"variable": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _plot_test_predictions(predictions: pd.DataFrame, task_name: str, path: Path) -> None:
    annual = predictions.groupby("annee", as_index=False)[["observe", "predit"]].sum()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].scatter(
        predictions["observe"],
        predictions["predit"],
        color="#315C4C",
        alpha=0.75,
        edgecolor="white",
    )
    maximum = float(predictions[["observe", "predit"]].to_numpy().max())
    axes[0].plot([0, maximum], [0, maximum], linestyle="--", color="#C96A3D")
    axes[0].set_xlabel("Valeur observée (m³ EBR)")
    axes[0].set_ylabel("Valeur prédite (m³ EBR)")
    axes[0].set_title("Observé contre prédit")
    axes[0].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

    axes[1].plot(annual["annee"], annual["observe"], marker="o", label="Observé", color="#9B5D2E")
    axes[1].plot(annual["annee"], annual["predit"], marker="o", label="Prédit", color="#315C4C")
    axes[1].set_xlabel("Année")
    axes[1].set_ylabel("Volume national (m³ EBR)")
    axes[1].set_title("Agrégation annuelle du test")
    axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)

    figure.suptitle(task_name.replace("_", " ").title())
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def train_task(task_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare les candidats, selectionne puis teste le meilleur modele d'une tache."""
    splits = load_task_splits(task_name)
    x_train, y_train = _split_xy(splits["train"])
    x_validation, y_validation = _split_xy(splits["validation"])
    x_test, y_test = _split_xy(splits["test"])

    validation_metrics: dict[str, dict[str, float]] = {}
    comparison_rows: list[dict[str, Any]] = []
    for model_name, estimator in candidate_models().items():
        pipeline = build_model_pipeline(x_train, estimator)
        pipeline.fit(x_train, y_train)
        predictions = predict_nonnegative(pipeline, x_validation)
        metrics = compute_metrics(y_validation, predictions)
        validation_metrics[model_name] = metrics
        comparison_rows.append(
            {"tache": task_name, "phase": "validation", "modele": model_name, **metrics}
        )

    best_name = select_best_model(validation_metrics)
    train_validation = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    x_train_validation, y_train_validation = _split_xy(train_validation)
    best_pipeline = build_model_pipeline(
        x_train_validation, candidate_models()[best_name]
    )
    best_pipeline.fit(x_train_validation, y_train_validation)
    test_predictions = predict_nonnegative(best_pipeline, x_test)
    test_metrics = compute_metrics(y_test, test_predictions)
    comparison_rows.append(
        {"tache": task_name, "phase": "test", "modele": best_name, **test_metrics}
    )

    model_dir = PROJECT_ROOT / f"models/{task_name}"
    metrics_dir = PROJECT_ROOT / "reports/metrics"
    figure_dir = PROJECT_ROOT / "reports/figures/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_dir / "best_model.joblib")

    prediction_frame = pd.DataFrame(
        {
            "annee": x_test["annee"].to_numpy(),
            "region": x_test["region"].to_numpy(),
            "observe": y_test.to_numpy(),
            "predit": test_predictions,
        }
    )
    prediction_frame["erreur"] = prediction_frame["predit"] - prediction_frame["observe"]
    prediction_frame["erreur_absolue"] = prediction_frame["erreur"].abs()
    prediction_frame.to_csv(
        metrics_dir / f"predictions_{task_name}.csv", sep=";", index=False
    )

    importance = _extract_importance(best_pipeline)
    importance.to_csv(
        metrics_dir / f"importance_{task_name}.csv", sep=";", index=False
    )
    _plot_test_predictions(
        prediction_frame,
        task_name,
        figure_dir / f"predictions_{task_name}.png",
    )

    task_report: dict[str, Any] = {
        "tache": task_name,
        "cible_originale": TASKS[task_name]["target"],
        "critere_selection": "wape_pct minimal sur validation, puis mae",
        "transformation_cible": "log1p pendant l'apprentissage, expm1 pour les predictions",
        "meilleur_modele": best_name,
        "validation": validation_metrics,
        "test": test_metrics,
        "nombre_lignes": {
            name: int(len(frame)) for name, frame in splits.items()
        },
        "principales_variables": importance.head(15).to_dict(orient="records"),
    }
    with (metrics_dir / f"resultats_{task_name}.json").open("w", encoding="utf-8") as file:
        json.dump(task_report, file, ensure_ascii=False, indent=2)
    return task_report, comparison_rows


def run_training(dataset: str = "tous") -> dict[str, Any]:
    """Entraine toutes les taches demandees et produit le rapport global."""
    if dataset not in {"demande", "offre", "tous"}:
        raise ValueError("dataset doit etre 'demande', 'offre' ou 'tous'")
    selected_tasks = [
        task_name
        for task_name, config in TASKS.items()
        if dataset == "tous" or config["dataset"] == dataset
    ]

    reports: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for task_name in selected_tasks:
        task_report, task_rows = train_task(task_name)
        reports[task_name] = task_report
        comparison_rows.extend(task_rows)

    metrics_dir = PROJECT_ROOT / "reports/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(
        metrics_dir / "comparaison_modeles.csv", sep=";", index=False
    )
    global_report = {
        "protocole": {
            "selection": "comparaison sur validation 2019-2021",
            "evaluation_finale": "meilleur modele reajuste sur 2000-2021 puis teste sur 2022-2024",
            "metriques": ["mae", "rmse", "r2", "wape_pct"],
            "random_state": RANDOM_STATE,
        },
        "taches": reports,
    }
    with (metrics_dir / "resultats_modeles.json").open("w", encoding="utf-8") as file:
        json.dump(global_report, file, ensure_ascii=False, indent=2)
    return global_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("demande", "offre", "tous"),
        default="tous",
        help="Modeles a entrainer (defaut: tous).",
    )
    args = parser.parse_args()
    report = run_training(args.dataset)
    for task_name, result in report["taches"].items():
        print(
            f"{task_name}: meilleur={result['meilleur_modele']}, "
            f"WAPE test={result['test']['wape_pct']:.2f} %, "
            f"R2 test={result['test']['r2']:.3f}"
        )


if __name__ == "__main__":
    main()
