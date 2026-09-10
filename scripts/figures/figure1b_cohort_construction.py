#!/usr/bin/env python
"""Figure 1b. Stepwise construction of the final analytical cohort."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *

RAW = DATA / "raw"
DERIVED = DATA / "derived" / "pf8_sea_artemisinin_clean.tsv"


def canonical_column(df, *names):
    cols = {str(c).strip(): c for c in df.columns}
    for name in names:
        if name in cols:
            return cols[name]
        key = name.lower()
        for col_name, col_value in cols.items():
            if str(col_name).lower() == key:
                return col_value
    return None


def compute_counts():
    required = [
        RAW / "Pf8_samples.txt",
        RAW / "Pf8_fws.tsv",
        RAW / "Pf8_drug_resistance_marker_genotypes.tsv",
        RAW / "Pf8_inferred_resistance_status_classification.tsv",
    ]
    if all(p.exists() for p in required):
        s = pd.read_csv(required[0], sep="\t", low_memory=False)
        f = pd.read_csv(required[1], sep="\t", low_memory=False)
        m = pd.read_csv(required[2], sep="\t", low_memory=False)
        r = pd.read_csv(required[3], sep="\t", low_memory=False)

        sample_cols = {
            "samples": canonical_column(s, "Sample", "sample"),
            "markers": canonical_column(m, "Sample", "sample"),
            "resistance": canonical_column(r, "Sample", "sample"),
            "fws": canonical_column(f, "Sample", "sample"),
        }
        s = s.rename(columns={sample_cols["samples"]: "Sample"})
        m = m.rename(columns={sample_cols["markers"]: "Sample"})
        r = r.rename(columns={sample_cols["resistance"]: "Sample"})
        f = f.rename(columns={sample_cols["fws"]: "Sample"})

        s = s[s["QC pass"].eq(True)].copy()
        matched = (
            s.merge(m, on="Sample", how="inner", validate="one_to_one")
             .merge(r, on="Sample", how="inner", validate="one_to_one")
             .merge(f, on="Sample", how="inner", validate="one_to_one")
        )
        matched_n = len(matched)
        sea = matched[matched["Population"].isin(["AS-SE-E", "AS-SE-W"])].copy()
        eligible_n = len(sea)
        y = sea["Artemisinin"].astype(str).str.strip().str.lower().map({"resistant": 1, "sensitive": 0})
        excluded_n = int(y.isna().sum())
        final_n = int(y.notna().sum())
        return matched_n, eligible_n, excluded_n, final_n
    if DERIVED.exists():
        final_n = len(pd.read_csv(DERIVED, sep="\t", low_memory=False))
        print("[WARNING] Full raw cohort tables not found; using manuscript checkpoint counts.")
        return 24409, 7768, 698, final_n
    raise SystemExit("No raw Pf8 resources or derived analytical table found.")


matched_n, eligible_n, excluded_n, final_n = compute_counts()

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis("off")

box_style = dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#2f3b52", linewidth=1.2)
arrow_style = dict(arrowstyle="-|>", color="#4a5a6a", lw=1.5)

# Source tables
source_x = 0.05
source_y = 0.78
source_w = 0.18
source_h = 0.12
source_labels = [
    "Pf8_samples.txt\nQC pass",
    "Pf8_drug_resistance_marker_genotypes.tsv\nmarkers",
    "Pf8_inferred_resistance_status_classification.tsv\nresistance labels",
    "Pf8_fws.tsv\nFws",
]
for i, label in enumerate(source_labels):
    x = source_x + (i % 2) * 0.18
    y = source_y - (i // 2) * 0.24
    ax.text(x, y, label, ha="center", va="center", fontsize=9.5,
            bbox=box_style)

# Merge and filtering boxes
step_positions = [
    (0.43, 0.75, "QC-pass sample set\nn = 24,409"),
    (0.62, 0.75, "Matched records\ninner merge on Sample\nn = 24,409"),
    (0.80, 0.75, "AS-SE-E / AS-SE-W\nSoutheast Asian populations\nn = 7,768"),
    (0.62, 0.36, "Undetermined artemisinin status\nexcluded\nn = 698"),
    (0.86, 0.36, "Final analytical cohort\nmodel-ready dataset\nn = 7,070"),
]
for x, y, label in step_positions:
    ax.text(x, y, label, ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef6ff", edgecolor="#4d7bb8", linewidth=1.2))

# Arrows
ax.annotate("", xy=(0.33, 0.82), xytext=(0.23, 0.82), arrowprops=arrow_style)
ax.annotate("", xy=(0.53, 0.82), xytext=(0.42, 0.82), arrowprops=arrow_style)
ax.annotate("", xy=(0.71, 0.82), xytext=(0.61, 0.82), arrowprops=arrow_style)
ax.annotate("", xy=(0.80, 0.57), xytext=(0.80, 0.68), arrowprops=arrow_style)
ax.annotate("", xy=(0.75, 0.36), xytext=(0.69, 0.36), arrowprops=arrow_style)
ax.annotate("", xy=(0.62, 0.46), xytext=(0.62, 0.64), arrowprops=arrow_style)

# Explanatory text
ax.text(0.50, 0.12,
        "Final cohort construction: keep QC-pass samples; merge genotype, resistance-label, and Fws tables;\n"
        "restrict to AS-SE-E / AS-SE-W populations; drop undetermined artemisinin status; retain the labeled cohort used for modeling.",
        ha="center", va="center", fontsize=10)

fig.suptitle("Final cohort generation from the Pf8 source tables", fontsize=14, y=0.96)
save(fig, "Figure_1b_cohort_construction.png")
