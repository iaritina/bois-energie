"""Calcule les indicateurs regionaux offre-demande pour 2025-2030."""

from __future__ import annotations

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

from src.forecasting.forecast_2030 import run_forecasts


BALANCE_PATH = (
    PROJECT_ROOT / "data/processed/forecasts/bilan_offre_demande_2025_2030.csv"
)

COVERAGE_THRESHOLDS = {
    "critique_max_exclu": 50.0,
    "elevee_max_exclu": 75.0,
    "moderee_max_exclu": 95.0,
    "equilibre_max_inclus": 105.0,
}

VULNERABILITY_COLORS = {
    "critique": "#9C2F2F",
    "elevee": "#D0603C",
    "moderee": "#E2A23A",
    "equilibre": "#87A96B",
    "surplus": "#28705A",
}


def classify_coverage(coverage_pct: float) -> str:
    """Classe un taux de couverture selon les seuils documentes."""
    if coverage_pct < COVERAGE_THRESHOLDS["critique_max_exclu"]:
        return "critique"
    if coverage_pct < COVERAGE_THRESHOLDS["elevee_max_exclu"]:
        return "elevee"
    if coverage_pct < COVERAGE_THRESHOLDS["moderee_max_exclu"]:
        return "moderee"
    if coverage_pct <= COVERAGE_THRESHOLDS["equilibre_max_inclus"]:
        return "equilibre"
    return "surplus"


def load_balance() -> pd.DataFrame:
    """Charge le bilan previsionnel et le regenere s'il est absent."""
    if not BALANCE_PATH.exists():
        run_forecasts()
    return pd.read_csv(BALANCE_PATH, sep=";")


def _linear_trend(years: pd.Series, values: pd.Series) -> float:
    if len(years) < 2:
        return 0.0
    return float(np.polyfit(years.to_numpy(dtype=float), values.to_numpy(dtype=float), 1)[0])


def compute_regional_indicators(balance: pd.DataFrame) -> pd.DataFrame:
    """Agrege les six annees et classe les regions par vulnerabilite."""
    required = {
        "annee",
        "region",
        "demande_totale_m3_ebr_prevue",
        "offre_totale_m3_ebr_prevue",
        "ecart_offre_demande_m3_ebr",
        "taux_couverture_pct",
        "statut",
    }
    missing = sorted(required.difference(balance.columns))
    if missing:
        raise ValueError(f"Colonnes manquantes dans le bilan: {missing}")

    rows: list[dict[str, Any]] = []
    for region, region_data in balance.groupby("region"):
        region_data = region_data.sort_values("annee")
        demand = region_data["demande_totale_m3_ebr_prevue"]
        supply = region_data["offre_totale_m3_ebr_prevue"]
        gap = region_data["ecart_offre_demande_m3_ebr"]
        deficit = (-gap).clip(lower=0)
        surplus = gap.clip(lower=0)
        demand_total = float(demand.sum())
        supply_total = float(supply.sum())
        global_coverage = supply_total / demand_total * 100
        deficit_years = region_data.loc[gap < 0, "annee"]

        rows.append(
            {
                "region": region,
                "demande_cumulee_m3_ebr": demand_total,
                "offre_cumulee_m3_ebr": supply_total,
                "ecart_cumule_m3_ebr": float(gap.sum()),
                "deficit_cumule_m3_ebr": float(deficit.sum()),
                "surplus_cumule_m3_ebr": float(surplus.sum()),
                "taux_couverture_global_pct": global_coverage,
                "taux_couverture_moyen_annuel_pct": float(
                    region_data["taux_couverture_pct"].mean()
                ),
                "taux_couverture_min_pct": float(
                    region_data["taux_couverture_pct"].min()
                ),
                "taux_couverture_max_pct": float(
                    region_data["taux_couverture_pct"].max()
                ),
                "deficit_max_annuel_m3_ebr": float(deficit.max()),
                "ecart_moyen_annuel_m3_ebr": float(gap.mean()),
                "annees_en_deficit": int((gap < 0).sum()),
                "annees_en_surplus": int((gap > 0).sum()),
                "premiere_annee_deficit": (
                    int(deficit_years.min()) if not deficit_years.empty else None
                ),
                "tendance_ecart_m3_ebr_par_an": _linear_trend(
                    region_data["annee"], gap
                ),
                "variation_couverture_2025_2030_points": float(
                    region_data.iloc[-1]["taux_couverture_pct"]
                    - region_data.iloc[0]["taux_couverture_pct"]
                ),
                "niveau_vulnerabilite": classify_coverage(global_coverage),
            }
        )

    indicators = pd.DataFrame(rows)
    indicators = indicators.sort_values(
        ["taux_couverture_global_pct", "deficit_cumule_m3_ebr"],
        ascending=[True, False],
    ).reset_index(drop=True)
    indicators.insert(0, "rang_vulnerabilite", np.arange(1, len(indicators) + 1))
    return indicators


def build_national_summary(
    balance: pd.DataFrame, indicators: pd.DataFrame
) -> dict[str, Any]:
    """Construit les indicateurs nationaux et les comptes par niveau."""
    total_demand = float(balance["demande_totale_m3_ebr_prevue"].sum())
    total_supply = float(balance["offre_totale_m3_ebr_prevue"].sum())
    level_counts = indicators["niveau_vulnerabilite"].value_counts()
    return {
        "demande_cumulee_m3_ebr": total_demand,
        "offre_cumulee_m3_ebr": total_supply,
        "ecart_cumule_m3_ebr": total_supply - total_demand,
        "taux_couverture_global_pct": total_supply / total_demand * 100,
        "regions_par_niveau": {
            level: int(level_counts.get(level, 0))
            for level in VULNERABILITY_COLORS
        },
        "region_plus_vulnerable": str(indicators.iloc[0]["region"]),
        "region_meilleure_couverture": str(indicators.iloc[-1]["region"]),
    }


def _plot_coverage_ranking(indicators: pd.DataFrame, output_dir: Path) -> None:
    ordered = indicators.sort_values("taux_couverture_global_pct", ascending=True)
    colors = [VULNERABILITY_COLORS[level] for level in ordered["niveau_vulnerabilite"]]
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.barh(ordered["region"], ordered["taux_couverture_global_pct"], color=colors)
    axis.axvspan(95, 105, color="#87A96B", alpha=0.12, label="Zone d'équilibre")
    axis.axvline(100, color="#315C4C", linestyle="--", linewidth=1.5)
    axis.set_xlabel("Taux de couverture global 2025–2030 (%)")
    axis.set_title("Classement régional de couverture offre–demande")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        output_dir / "classement_couverture_regions.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _plot_cumulative_deficits(indicators: pd.DataFrame, output_dir: Path) -> None:
    deficits = indicators.loc[indicators["deficit_cumule_m3_ebr"] > 0].copy()
    deficits = deficits.sort_values("deficit_cumule_m3_ebr", ascending=True)
    figure, axis = plt.subplots(figsize=(11, 7.5))
    axis.barh(deficits["region"], deficits["deficit_cumule_m3_ebr"], color="#C9553D")
    axis.set_xlabel("Déficit cumulé 2025–2030 (m³ EBR)")
    axis.set_title("Volume cumulé non couvert par région")
    axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "deficits_cumules_regions.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _plot_annual_coverage(balance: pd.DataFrame, output_dir: Path) -> None:
    pivot = balance.pivot(index="region", columns="annee", values="taux_couverture_pct")
    figure, axis = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        center=100,
        vmin=0,
        vmax=max(150, float(pivot.quantile(0.95).max())),
        ax=axis,
        cbar_kws={"label": "Taux de couverture (%)"},
    )
    axis.set_title("Évolution annuelle de la couverture régionale")
    axis.set_xlabel("Année")
    axis.set_ylabel("Région")
    figure.tight_layout()
    figure.savefig(
        output_dir / "couverture_region_annee.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _write_markdown_report(
    indicators: pd.DataFrame, national: dict[str, Any], path: Path
) -> None:
    lines = [
        "# Indicateurs régionaux offre–demande 2025–2030",
        "",
        "> Résultats expérimentaux construits à partir de données et covariables synthétiques.",
        "",
        "## Synthèse nationale",
        "",
        f"- Taux de couverture global : {national['taux_couverture_global_pct']:.2f} %",
        f"- Écart cumulé : {national['ecart_cumule_m3_ebr'] / 1_000_000:.2f} millions de m³ EBR",
        f"- Région la plus vulnérable : {national['region_plus_vulnerable']}",
        f"- Meilleure couverture : {national['region_meilleure_couverture']}",
        "",
        "## Seuils",
        "",
        "- Critique : couverture inférieure à 50 %.",
        "- Élevée : de 50 % à moins de 75 %.",
        "- Modérée : de 75 % à moins de 95 %.",
        "- Équilibre : de 95 % à 105 %.",
        "- Surplus : supérieure à 105 %.",
        "",
        "## Dix régions les plus vulnérables",
        "",
        "| Rang | Région | Couverture | Déficit cumulé | Niveau |",
        "|---:|---|---:|---:|---|",
    ]
    for row in indicators.head(10).itertuples():
        lines.append(
            f"| {row.rang_vulnerabilite} | {row.region} | "
            f"{row.taux_couverture_global_pct:.2f} % | "
            f"{row.deficit_cumule_m3_ebr:,.0f} m³ EBR | "
            f"{row.niveau_vulnerabilite} |"
        )
    lines.extend(
        [
            "",
            "Les niveaux décrivent un scénario synthétique et ne constituent pas un diagnostic officiel. Les seuils sont des règles de lecture configurables, pas des normes réglementaires.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_regional_indicators() -> dict[str, Any]:
    """Calcule, sauvegarde et visualise les indicateurs regionaux."""
    balance = load_balance()
    indicators = compute_regional_indicators(balance)
    national = build_national_summary(balance, indicators)

    generated_dir = PROJECT_ROOT / "reports/generated"
    figure_dir = PROJECT_ROOT / "reports/figures/indicators"
    generated_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    indicators.to_csv(
        generated_dir / "indicateurs_regionaux_2025_2030.csv", sep=";", index=False
    )
    indicators.sort_values("deficit_cumule_m3_ebr", ascending=False).to_csv(
        generated_dir / "classement_deficits_regionaux_2025_2030.csv",
        sep=";",
        index=False,
    )
    report = {
        "nature": "indicateurs experimentaux sur donnees synthetiques",
        "periode": [2025, 2030],
        "seuils_couverture": COVERAGE_THRESHOLDS,
        "synthese_nationale": national,
        "regions": json.loads(indicators.to_json(orient="records")),
    }
    with (generated_dir / "indicateurs_regionaux_2025_2030.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    _write_markdown_report(
        indicators,
        national,
        generated_dir / "rapport_indicateurs_regionaux.md",
    )

    _plot_coverage_ranking(indicators, figure_dir)
    _plot_cumulative_deficits(indicators, figure_dir)
    _plot_annual_coverage(balance, figure_dir)
    return report


def main() -> None:
    report = run_regional_indicators()
    national = report["synthese_nationale"]
    print(
        f"Couverture nationale 2025-2030: "
        f"{national['taux_couverture_global_pct']:.2f} %"
    )
    print(f"Region la plus vulnerable: {national['region_plus_vulnerable']}")
    print(f"Meilleure couverture: {national['region_meilleure_couverture']}")


if __name__ == "__main__":
    main()
