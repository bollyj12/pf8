#!/usr/bin/env python
"""
visualize_pf8_results.py
========================

Matplotlib-based visualization and table-generation utility for the Pf8
artemisinin partial-resistance reproducibility project.

The script reads:
  - the final analytical dataset
  - nested-CV summary / bootstrap CI results
  - LOCO results
  - SHAP source-variable summaries

It generates:
  1. console data-structure report
  2. data_structure_overview.png
  3. country_sample_counts.png
  4. country_resistance_prevalence.png
  5. nested_cv_auc_ci.png
  6. nested_cv_performance_table.png
  7. nested_cv_performance_table.csv
  8. loco_auc.png
  9. loco_pr_auc.png
 10. loco_performance_table.png
 11. loco_performance_table.csv
 12. shap_xgboost_top_features.png
 13. shap_randomforest_top_features.png
 14. cohort_summary_table.png
 15. cohort_summary_table.csv

Usage from the project root
---------------------------
python visualize_pf8_results.py

Or specify paths:
python visualize_pf8_results.py ^
  --data data/derived/pf8_sea_artemisinin_clean.tsv ^
  --results results ^
  --outdir figures_tables

Only matplotlib is used for plotting. No seaborn is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET = "artemisinin_resistant"


def find_first(base: Path, patterns: list[str]) -> Path | None:
    """Return first matching file from any supplied glob pattern."""
    for pattern in patterns:
        matches = sorted(base.rglob(pattern))
        if matches:
            return matches[0]
    return None


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED FIGURE] {path}")


def save_table_png(df: pd.DataFrame, title: str, path: Path,
                   decimals: int = 3, font_size: int = 9):
    """Render a DataFrame as a standalone matplotlib table figure."""
    display = df.copy()

    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}"
            )

    nrows, ncols = display.shape
    width = max(8.5, ncols * 1.35)
    height = max(2.2, 1.0 + (nrows + 1) * 0.42)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.35)

    # Bold header without imposing custom colors/styles.
    for col in range(ncols):
        table[(0, col)].set_text_props(weight="bold")

    save_figure(fig, path)


def report_data_structure(df: pd.DataFrame):
    print("\n" + "=" * 78)
    print("Pf8 ANALYTICAL DATA STRUCTURE")
    print("=" * 78)
    print(f"Rows:                 {len(df):,}")
    print(f"Columns:              {len(df.columns):,}")
    print(f"Duplicate Sample IDs: {df['Sample'].duplicated().sum() if 'Sample' in df else 'N/A'}")
    print(f"Missing cells:        {int(df.isna().sum().sum()):,}")

    if "Study" in df:
        print(f"Studies:              {df['Study'].nunique():,}")
    if "Country" in df:
        print(f"Countries:            {df['Country'].nunique():,}")

    if TARGET in df:
        resistant = int((df[TARGET] == 1).sum())
        sensitive = int((df[TARGET] == 0).sum())
        print(f"Resistant:            {resistant:,}")
        print(f"Sensitive:            {sensitive:,}")

    numeric = list(df.select_dtypes(include=np.number).columns)
    categorical = [c for c in df.columns if c not in numeric]

    print(f"Numeric columns:      {len(numeric)}")
    print(f"Non-numeric columns:  {len(categorical)}")

    print("\nCOLUMN STRUCTURE")
    struct = pd.DataFrame({
        "Column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    with pd.option_context("display.max_rows", 200, "display.max_colwidth", 40):
        print(struct.to_string(index=False))

    print("=" * 78 + "\n")
    return struct


def make_data_structure_figure(df: pd.DataFrame, outdir: Path):
    metadata_names = {
        "Sample", "Study", "Country", "Population", "Year",
        "Country latitude", "Country longitude",
        "Admin level 1 latitude", "Admin level 1 longitude"
    }

    target_cols = [c for c in df.columns if c == TARGET or c.lower() == "artemisinin"]
    numeric_cols = [c for c in ["Year", "Fws"] if c in df.columns]
    kelch_cols = [c for c in df.columns if "kelch13" in c.lower()]

    marker_cols = [
        c for c in df.columns
        if c not in metadata_names
        and c not in target_cols
        and c not in numeric_cols
        and c not in kelch_cols
    ]

    counts = pd.Series({
        "Metadata/grouping": len([c for c in df.columns if c in metadata_names]),
        "Curated marker variables": len(marker_cols),
        "Numeric model variables": len(numeric_cols),
        "Kelch13 archival fields": len(kelch_cols),
        "Outcome fields": len(target_cols),
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Number of source columns")
    ax.set_title("Structure of the Pf8 Analytical Dataset")
    ax.tick_params(axis="x", rotation=25)
    for i, value in enumerate(counts.values):
        ax.text(i, value, str(int(value)), ha="center", va="bottom")
    fig.tight_layout()
    save_figure(fig, outdir / "data_structure_overview.png")


def make_cohort_outputs(df: pd.DataFrame, outdir: Path):
    if "Country" not in df.columns:
        return

    cohort = df.groupby("Country").size().rename("N").reset_index()

    if TARGET in df.columns:
        res = (
            df.groupby("Country")[TARGET]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "Resistant", "count": "Total"})
        )
        res["Sensitive"] = res["Total"] - res["Resistant"]
        res["Resistant_pct"] = 100 * res["Resistant"] / res["Total"]
        cohort = cohort.merge(
            res[["Country", "Resistant", "Sensitive", "Resistant_pct"]],
            on="Country",
            how="left",
        )

    cohort = cohort.sort_values("N", ascending=False)
    cohort.to_csv(outdir / "cohort_summary_table.csv", index=False)
    print(f"[SAVED TABLE] {outdir / 'cohort_summary_table.csv'}")

    save_table_png(
        cohort,
        "Country-level Pf8 analytical cohort",
        outdir / "cohort_summary_table.png",
        decimals=1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cohort["Country"], cohort["N"])
    ax.set_ylabel("Number of samples")
    ax.set_title("Pf8 Southeast Asian Analytical Cohort")
    for i, n in enumerate(cohort["N"]):
        ax.text(i, n, f"{int(n):,}", ha="center", va="bottom")
    fig.tight_layout()
    save_figure(fig, outdir / "country_sample_counts.png")

    if "Resistant_pct" in cohort.columns:
        prevalence = cohort.sort_values("Resistant_pct", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(prevalence["Country"], prevalence["Resistant_pct"])
        ax.set_ylabel("Pf8-inferred resistant class (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Country-level Prevalence of Pf8-inferred Artemisinin Partial Resistance")
        for i, value in enumerate(prevalence["Resistant_pct"]):
            ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom")
        fig.tight_layout()
        save_figure(fig, outdir / "country_resistance_prevalence.png")


def load_nested_summary(results: Path) -> pd.DataFrame | None:
    files = sorted(results.rglob("nested_*_summary.csv"))
    rows = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if {"ROC_AUC", "PR_AUC", "Set", "Model"}.issubset(d.columns):
                rows.append(d.iloc[[0]].copy())
        except Exception:
            pass

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def make_nested_outputs(results: Path, outdir: Path):
    summary = load_nested_summary(results)
    if summary is None or summary.empty:
        print("[SKIP] No nested model summary CSVs found.")
        return

    cols = [
        c for c in [
            "Set", "Model", "ROC_AUC", "PR_AUC", "MCC",
            "Balanced_Accuracy", "Accuracy", "Precision",
            "Sensitivity", "Specificity", "F1", "N"
        ]
        if c in summary.columns
    ]

    table = summary[cols].copy()
    table.to_csv(outdir / "nested_cv_performance_table.csv", index=False)
    print(f"[SAVED TABLE] {outdir / 'nested_cv_performance_table.csv'}")
    save_table_png(
        table,
        "Nested study-grouped out-of-fold model performance",
        outdir / "nested_cv_performance_table.png",
        decimals=3,
        font_size=8,
    )

    ci_file = find_first(
        results,
        [
            "nested_primary_model_CI_2000.csv",
            "*primary*CI*2000*.csv",
        ],
    )

    if ci_file:
        ci = pd.read_csv(ci_file)
        if {"Label", "ROC_AUC", "CI_low", "CI_high"}.issubset(ci.columns):
            ci = ci.sort_values("ROC_AUC")
            lower = ci["ROC_AUC"] - ci["CI_low"]
            upper = ci["CI_high"] - ci["ROC_AUC"]

            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.errorbar(
                ci["ROC_AUC"],
                np.arange(len(ci)),
                xerr=np.vstack([lower, upper]),
                fmt="o",
                capsize=4,
            )
            ax.set_yticks(np.arange(len(ci)))
            ax.set_yticklabels(ci["Label"])
            ax.set_xlabel("ROC-AUC")
            ax.set_title("Nested Study-grouped Model Discrimination\n95% Study-cluster Bootstrap Confidence Intervals")
            ax.set_xlim(max(0, ci["CI_low"].min() - 0.03), 1.0)
            fig.tight_layout()
            save_figure(fig, outdir / "nested_cv_auc_ci.png")


def load_loco(results: Path) -> pd.DataFrame | None:
    files = sorted(results.rglob("loco_tuned_*XGBoost.csv"))
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if "Country" in d.columns and "ROC_AUC" in d.columns:
                if "Set" not in d.columns:
                    if "ExtendedStrict" in f.name:
                        d["Set"] = "ExtendedStrict"
                    else:
                        d["Set"] = "Full"
                frames.append(d)
        except Exception:
            pass
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def make_loco_outputs(df: pd.DataFrame, results: Path, outdir: Path):
    loco = load_loco(results)
    if loco is None or loco.empty:
        print("[SKIP] No LOCO result CSVs found.")
        return

    # Add observed country prevalence from the analytical dataset.
    if "Country" in df.columns and TARGET in df.columns:
        prev = (
            df.groupby("Country")[TARGET]
            .agg(N="size", Resistant="sum", Prevalence="mean")
            .reset_index()
        )
        prev["Resistant_pct"] = prev["Prevalence"] * 100
        loco = loco.drop(columns=["N"], errors="ignore").merge(
            prev[["Country", "N", "Resistant", "Resistant_pct"]],
            on="Country",
            how="left",
        )

    table_cols = [
        c for c in [
            "Set", "Country", "N", "Resistant", "Resistant_pct",
            "ROC_AUC", "PR_AUC", "MCC", "Balanced_Accuracy",
            "Sensitivity", "Specificity", "F1"
        ]
        if c in loco.columns
    ]
    table = loco[table_cols].sort_values(["Set", "Country"])
    table.to_csv(outdir / "loco_performance_table.csv", index=False)
    print(f"[SAVED TABLE] {outdir / 'loco_performance_table.csv'}")
    save_table_png(
        table,
        "Leave-one-country-out geographical transportability",
        outdir / "loco_performance_table.png",
        decimals=3,
        font_size=7,
    )

    # ROC-AUC figure
    pivot = loco.pivot_table(
        index="Country", columns="Set", values="ROC_AUC", aggfunc="first"
    )
    countries = list(pivot.index)
    sets = list(pivot.columns)
    x = np.arange(len(countries))
    width = 0.8 / max(1, len(sets))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, setname in enumerate(sets):
        offset = (i - (len(sets) - 1) / 2) * width
        vals = pivot[setname].to_numpy()
        bars = ax.bar(x + offset, vals, width=width, label=setname)
        for bar, value in zip(bars, vals):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Leave-one-country-out ROC-AUC")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, outdir / "loco_auc.png")

    # PR-AUC figure, shown separately rather than subplot.
    if "PR_AUC" in loco.columns:
        full = loco[loco["Set"].astype(str).str.lower().eq("full")].copy()
        if full.empty:
            full = loco.copy()

        full = full.sort_values("Country")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        bars = ax.bar(full["Country"], full["PR_AUC"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("PR-AUC / Average precision")
        ax.set_title("Leave-one-country-out PR-AUC")
        for bar, value in zip(bars, full["PR_AUC"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )
        fig.tight_layout()
        save_figure(fig, outdir / "loco_pr_auc.png")


def make_shap_plot(file: Path, title: str, path: Path, top_n: int = 12):
    d = pd.read_csv(file)
    required = {"Source_variable", "Mean_abs_SHAP"}
    if not required.issubset(d.columns):
        print(f"[SKIP] Invalid SHAP file: {file}")
        return

    top = d.nlargest(top_n, "Mean_abs_SHAP").sort_values("Mean_abs_SHAP")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(top["Source_variable"], top["Mean_abs_SHAP"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, path)


def make_shap_outputs(results: Path, outdir: Path):
    xgb = find_first(results, ["XGBoost_SHAP_source.csv", "*XGBoost*SHAP*source*.csv"])
    rf = find_first(results, ["RF_SHAP_source.csv", "*RF*SHAP*source*.csv"])

    if xgb:
        make_shap_plot(
            xgb,
            "XGBoost Global Predictive Attribution\nTop Source Variables",
            outdir / "shap_xgboost_top_features.png",
        )
    else:
        print("[SKIP] XGBoost SHAP summary not found.")

    if rf:
        make_shap_plot(
            rf,
            "Random Forest Global Predictive Attribution\nTop Source Variables",
            outdir / "shap_randomforest_top_features.png",
        )
    else:
        print("[SKIP] Random Forest SHAP summary not found.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate matplotlib figures and tables for the Pf8 analysis."
    )
    parser.add_argument(
        "--data",
        default="data/derived/pf8_sea_artemisinin_clean.tsv",
        help="Path to the final analytical TSV.",
    )
    parser.add_argument(
        "--results",
        default="results",
        help="Root directory containing analysis result CSVs.",
    )
    parser.add_argument(
        "--outdir",
        default="figures_tables",
        help="Output directory.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    results_path = Path(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise SystemExit(
            f"Analytical dataset not found: {data_path}\n"
            "Run the script from the project root or pass --data."
        )

    print(f"[LOAD] {data_path}")
    df = pd.read_csv(data_path, sep="\t", low_memory=False)

    structure = report_data_structure(df)
    structure.to_csv(outdir / "data_structure.csv", index=False)
    print(f"[SAVED TABLE] {outdir / 'data_structure.csv'}")

    make_data_structure_figure(df, outdir)
    make_cohort_outputs(df, outdir)
    make_nested_outputs(results_path, outdir)
    make_loco_outputs(df, results_path, outdir)
    make_shap_outputs(results_path, outdir)

    print("\n" + "=" * 78)
    print("FIGURE/TABLE GENERATION COMPLETE")
    print("=" * 78)
    print(f"Output directory: {outdir.resolve()}")
    for f in sorted(outdir.iterdir()):
        if f.is_file():
            print(" -", f.name)
    print("=" * 78)


if __name__ == "__main__":
    main()
