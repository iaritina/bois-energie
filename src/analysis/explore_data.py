"""Genere les rapports exploratoires de l'offre et de la demande historiques."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.clean_data import run_cleaning


DATASETS = {
    "demande": {
        "path": PROJECT_ROOT / "data/interim/demande_historique.csv",
        "targets": [
            "demande_bois_chauffe_m3_ebr",
            "demande_charbon_m3_ebr",
            "demande_totale_m3_ebr",
        ],
        "main_target": "demande_totale_m3_ebr",
        "title": "Demande",
    },
    "offre": {
        "path": PROJECT_ROOT / "data/interim/offre_historique.csv",
        "targets": [
            "production_bois_feu_m3_ebr",
            "production_bois_charbon_m3_ebr",
            "offre_totale_m3_ebr",
        ],
        "main_target": "offre_totale_m3_ebr",
        "title": "Offre",
    },
}

METADATA_COLUMNS = {
    "id_observation",
    "source_principale",
    "statut_donnee",
    "commentaire_collecte",
}

COLORS = ["#9B5D2E", "#D69E2E", "#315C4C"]


def load_historical(kind: str) -> pd.DataFrame:
    """Charge les donnees historiques et regenere l'intermediaire si necessaire."""
    if kind not in DATASETS:
        raise ValueError("kind doit etre 'demande' ou 'offre'")

    path = DATASETS[kind]["path"]
    if not path.exists():
        run_cleaning(kind)

    frame = pd.read_csv(path, sep=";")
    if "statut_donnee" in frame.columns:
        frame = frame.loc[frame["statut_donnee"] == "historique_synthetique"].copy()
    return frame


def compute_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Calcule les statistiques descriptives et indicateurs de qualite."""
    numeric = frame.select_dtypes(include="number")
    statistics = numeric.describe(
        percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
    ).transpose()
    statistics["missing_count"] = numeric.isna().sum()
    statistics["missing_pct"] = numeric.isna().mean().mul(100)
    statistics["zero_count"] = numeric.eq(0).sum()

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    statistics["iqr_outlier_candidates"] = (
        numeric.lt(lower) | numeric.gt(upper)
    ).sum()
    return statistics


def _strongest_correlations(
    correlations: pd.DataFrame, targets: list[str]
) -> dict[str, list[dict[str, float]]]:
    result: dict[str, list[dict[str, float]]] = {}
    excluded = set(targets) | {"annee"}
    for target in targets:
        if target not in correlations:
            continue
        values = correlations[target].drop(labels=list(excluded), errors="ignore").dropna()
        values = values.reindex(values.abs().sort_values(ascending=False).index).head(5)
        result[target] = [
            {"variable": variable, "correlation": round(float(value), 4)}
            for variable, value in values.items()
        ]
    return result


def build_summary(
    frame: pd.DataFrame, kind: str, statistics: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Construit le resume serialisable utilise par le rapport exploratoire."""
    if kind not in DATASETS:
        raise ValueError("kind doit etre 'demande' ou 'offre'")
    statistics = compute_statistics(frame) if statistics is None else statistics
    targets = DATASETS[kind]["targets"]
    correlations = frame.select_dtypes(include="number").corr()

    missing = frame.isna().mean().mul(100).sort_values(ascending=False)
    missing = missing[missing > 0]
    target_summary: dict[str, dict[str, float | None]] = {}
    for target in targets:
        values = frame[target]
        target_summary[target] = {
            "minimum": _finite_or_none(values.min()),
            "mediane": _finite_or_none(values.median()),
            "moyenne": _finite_or_none(values.mean()),
            "maximum": _finite_or_none(values.max()),
        }

    outliers = statistics["iqr_outlier_candidates"].sort_values(ascending=False)
    outliers = outliers[(outliers > 0) & ~outliers.index.isin(["annee"])]

    return {
        "jeu_donnees": kind,
        "observations": int(len(frame)),
        "nombre_regions": int(frame["region"].nunique()),
        "annee_min": int(frame["annee"].min()),
        "annee_max": int(frame["annee"].max()),
        "doublons_region_annee": int(frame.duplicated(["region", "annee"]).sum()),
        "valeurs_manquantes_pct": {
            str(column): round(float(value), 3) for column, value in missing.items()
        },
        "resume_cibles": target_summary,
        "candidats_valeurs_extremes_iqr": {
            str(column): int(value) for column, value in outliers.items()
        },
        "correlations_principales": _strongest_correlations(correlations, targets),
    }


def _finite_or_none(value: object) -> float | None:
    if pd.isna(value) or not np.isfinite(float(value)):
        return None
    return round(float(value), 4)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_missing_values(frame: pd.DataFrame, kind: str, output_dir: Path) -> None:
    missing = frame.isna().mean().mul(100).sort_values()
    missing = missing[(missing > 0) & ~missing.index.isin(METADATA_COLUMNS)]

    figure, axis = plt.subplots(figsize=(10, max(4, len(missing) * 0.32)))
    if missing.empty:
        axis.text(0.5, 0.5, "Aucune valeur manquante", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.barh(missing.index, missing.values, color="#C96A3D")
        axis.set_xlabel("Valeurs manquantes (%)")
        axis.set_xlim(0, max(5, float(missing.max()) * 1.15))
        axis.grid(axis="x", alpha=0.2)
    axis.set_title(f"{DATASETS[kind]['title']} — variables incomplètes")
    _save_figure(figure, output_dir / "valeurs_manquantes.png")


def plot_target_distributions(frame: pd.DataFrame, kind: str, output_dir: Path) -> None:
    targets = DATASETS[kind]["targets"]
    figure, axes = plt.subplots(1, len(targets), figsize=(16, 4.5))
    for axis, target, color in zip(axes, targets, COLORS):
        sns.histplot(frame[target].dropna(), bins=24, kde=True, ax=axis, color=color)
        axis.set_title(target.replace("_", " "))
        axis.set_xlabel("Volume (m³ EBR)")
        axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    figure.suptitle(f"{DATASETS[kind]['title']} — distribution des cibles", y=1.02)
    _save_figure(figure, output_dir / "distributions_cibles.png")


def plot_national_trend(frame: pd.DataFrame, kind: str, output_dir: Path) -> None:
    targets = DATASETS[kind]["targets"]
    annual = frame.groupby("annee", as_index=False)[targets].sum(min_count=1)
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for target, color in zip(targets, COLORS):
        axis.plot(
            annual["annee"],
            annual[target],
            label=target.replace("_", " "),
            color=color,
            linewidth=2.2,
        )
    axis.set_title(f"{DATASETS[kind]['title']} — évolution nationale historique")
    axis.set_xlabel("Année")
    axis.set_ylabel("Volume agrégé (m³ EBR)")
    axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _save_figure(figure, output_dir / "evolution_nationale.png")


def plot_regional_heatmap(frame: pd.DataFrame, kind: str, output_dir: Path) -> None:
    target = DATASETS[kind]["main_target"]
    pivot = frame.pivot(index="region", columns="annee", values=target)
    figure, axis = plt.subplots(figsize=(15, 8))
    sns.heatmap(
        pivot,
        cmap="YlOrBr",
        ax=axis,
        cbar_kws={"label": "Volume total (m³ EBR)"},
    )
    axis.set_title(f"{DATASETS[kind]['title']} totale par région et par année")
    axis.set_xlabel("Année")
    axis.set_ylabel("Région")
    _save_figure(figure, output_dir / "evolution_regions.png")


def plot_correlations(frame: pd.DataFrame, kind: str, output_dir: Path) -> None:
    numeric = frame.select_dtypes(include="number").drop(columns="annee", errors="ignore")
    correlations = numeric.corr()
    mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
    size = max(10, len(correlations) * 0.58)
    figure, axis = plt.subplots(figsize=(size, size * 0.82))
    sns.heatmap(
        correlations,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.25,
        cbar_kws={"label": "Corrélation de Pearson", "shrink": 0.75},
        ax=axis,
    )
    axis.set_title(f"{DATASETS[kind]['title']} — corrélations numériques")
    _save_figure(figure, output_dir / "correlations.png")


def _write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    kind = str(summary["jeu_donnees"])
    title = DATASETS[kind]["title"]
    lines = [
        f"# Analyse exploratoire — {title}",
        "",
        "## Couverture",
        "",
        f"- Observations : {summary['observations']}",
        f"- Régions : {summary['nombre_regions']}",
        f"- Période : {summary['annee_min']}–{summary['annee_max']}",
        f"- Doublons région-année : {summary['doublons_region_annee']}",
        "",
        "## Valeurs manquantes",
        "",
    ]
    missing = summary["valeurs_manquantes_pct"]
    if missing:
        lines.extend(f"- `{column}` : {value:.3f} %" for column, value in missing.items())
    else:
        lines.append("Aucune valeur manquante.")

    lines.extend(["", "## Corrélations principales avec les cibles", ""])
    for target, values in summary["correlations_principales"].items():
        lines.append(f"### `{target}`")
        lines.append("")
        lines.extend(
            f"- `{item['variable']}` : {item['correlation']:.4f}" for item in values
        )
        lines.append("")

    lines.extend(
        [
            "## Interprétation",
            "",
            "Les valeurs extrêmes détectées par la règle IQR sont des candidats à examiner, pas des erreurs automatiquement supprimées. Les corrélations décrivent des associations et ne démontrent pas une relation causale. Les variables directement utilisées pour calculer une cible devront être exclues de ses prédicteurs afin d'éviter une fuite de données.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_exploration(kind: str) -> dict[str, Any]:
    """Execute l'analyse exploratoire complete d'un jeu historique."""
    frame = load_historical(kind)
    statistics = compute_statistics(frame)
    summary = build_summary(frame, kind, statistics)

    generated_dir = PROJECT_ROOT / "reports/generated"
    figure_dir = PROJECT_ROOT / f"reports/figures/{kind}"
    generated_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    statistics.to_csv(generated_dir / f"statistiques_{kind}.csv", sep=";")
    frame.select_dtypes(include="number").corr().to_csv(
        generated_dir / f"correlations_{kind}.csv", sep=";"
    )
    with (generated_dir / f"analyse_{kind}.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    _write_markdown_report(summary, generated_dir / f"rapport_exploratoire_{kind}.md")

    sns.set_theme(style="whitegrid", context="notebook")
    plot_missing_values(frame, kind, figure_dir)
    plot_target_distributions(frame, kind, figure_dir)
    plot_national_trend(frame, kind, figure_dir)
    plot_regional_heatmap(frame, kind, figure_dir)
    plot_correlations(frame, kind, figure_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("demande", "offre", "tous"),
        default="tous",
        help="Jeu historique à analyser (défaut : tous).",
    )
    args = parser.parse_args()
    kinds = ("demande", "offre") if args.dataset == "tous" else (args.dataset,)

    for kind in kinds:
        summary = run_exploration(kind)
        print(
            f"{kind.capitalize()}: {summary['observations']} observations, "
            f"{summary['nombre_regions']} régions, "
            f"{summary['annee_min']}-{summary['annee_max']}"
        )


if __name__ == "__main__":
    main()
