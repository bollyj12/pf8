#!/usr/bin/env python
"""Figure 1. Cohort flow from Pf8 records to final analytical cohort."""
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
        RAW/"Pf8_samples.txt",
        RAW/"Pf8_fws.tsv",
        RAW/"Pf8_drug_resistance_marker_genotypes.tsv",
        RAW/"Pf8_inferred_resistance_status_classification.tsv",
    ]
    if all(p.exists() for p in required):
        s=pd.read_csv(required[0],sep="\t",low_memory=False)
        f=pd.read_csv(required[1],sep="\t",low_memory=False)
        m=pd.read_csv(required[2],sep="\t",low_memory=False)
        r=pd.read_csv(required[3],sep="\t",low_memory=False)

        sample_cols = {
            "samples": canonical_column(s, "Sample", "sample"),
            "markers": canonical_column(m, "Sample", "sample"),
            "resistance": canonical_column(r, "Sample", "sample"),
            "fws": canonical_column(f, "Sample", "sample"),
        }
        s=s.rename(columns={sample_cols["samples"]: "Sample"})
        m=m.rename(columns={sample_cols["markers"]: "Sample"})
        r=r.rename(columns={sample_cols["resistance"]: "Sample"})
        f=f.rename(columns={sample_cols["fws"]: "Sample"})

        s=s[s["QC pass"].eq(True)].copy()
        merged=(s.merge(m,on="Sample",how="inner",validate="one_to_one")
                 .merge(r,on="Sample",how="inner",validate="one_to_one")
                 .merge(f,on="Sample",how="inner",validate="one_to_one"))
        matched=len(merged)
        sea=merged[merged["Population"].isin(["AS-SE-E","AS-SE-W"])].copy()
        eligible=len(sea)
        y=sea["Artemisinin"].astype(str).str.strip().str.lower().map({"resistant":1,"sensitive":0})
        excluded=int(y.isna().sum())
        final=int(y.notna().sum())
        return matched, eligible, excluded, final
    if DERIVED.exists():
        final=len(pd.read_csv(DERIVED,sep="\t",low_memory=False))
        print("[WARNING] Complete raw source files not found; using manuscript checkpoints for upstream counts.")
        return 24409, 7768, 698, final
    raise SystemExit("No raw Pf8 resources or derived analytical table found.")

matched, eligible, excluded, final = compute_counts()

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.axis("off")
labels=[
    (0.12,0.75,f"Matched Pf8 records\nn = {matched:,}\nmetadata + markers + resistance + Fws"),
    (0.50,0.75,f"Southeast Asian populations\nAS-SE-E / AS-SE-W\nn = {eligible:,}"),
    (0.50,0.32,f"Excluded undetermined\nartemisinin classification\nn = {excluded:,}"),
    (0.88,0.75,f"Final analytical cohort\nn = {final:,}\n23 studies, 5 countries"),
]
for x,y,t in labels:
    ax.text(x,y,t,ha="center",va="center",
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="black"),
            fontsize=10)
ax.annotate("",xy=(0.39,0.75),xytext=(0.23,0.75),arrowprops=dict(arrowstyle="->"))
ax.annotate("",xy=(0.77,0.75),xytext=(0.61,0.75),arrowprops=dict(arrowstyle="->"))
ax.annotate("",xy=(0.50,0.43),xytext=(0.50,0.64),arrowprops=dict(arrowstyle="->"))
ax.text(0.5,0.08,"PfKelch13-derived outcome retained only for labeling; kelch13_349-726_ns_changes excluded from predictors.",
        ha="center",va="center",fontsize=9)
fig.suptitle("Cohort flow from Pf8 records to the final analytical cohort",fontsize=12)
save(fig,"Figure_1_cohort_flow.png")
