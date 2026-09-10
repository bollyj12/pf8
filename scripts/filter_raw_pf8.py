#!/usr/bin/env python
"""Reconstruct the 7,070-sample Southeast Asian Pf8 analytical cohort from raw Pf8 tables."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

RAW_NAMES = {
    "samples":"Pf8_samples.txt",
    "markers":"Pf8_drug_resistance_marker_genotypes.tsv",
    "resistance":"Pf8_inferred_resistance_status_classification.tsv",
    "fws":"Pf8_fws.tsv",
}

def read_tsv(path):
    return pd.read_csv(path,sep="\t",low_memory=False)

def canonical_column(df, *names):
    cols={str(c).strip(): c for c in df.columns}
    for name in names:
        if name in cols:
            return cols[name]
        key=name.lower()
        for col_name, col_value in cols.items():
            if str(col_name).lower() == key:
                return col_value
    return None

def require(df, cols, name):
    miss=[c for c in cols if canonical_column(df, c) is None]
    if miss:
        raise ValueError(f"{name}: missing columns {miss}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw-dir",default="data/raw")
    ap.add_argument("--out",default="data/derived/pf8_sea_artemisinin_clean.tsv")
    args=ap.parse_args()
    raw=Path(args.raw_dir)
    paths={k:raw/v for k,v in RAW_NAMES.items()}
    missing=[str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("Missing raw files:\n  "+"\n  ".join(missing)+"\nRun: python scripts/download_raw_pf8.py")

    print("="*72)
    print("Pf8 RAW DATA FILTERING / COHORT RECONSTRUCTION")
    print("="*72)
    samples=read_tsv(paths["samples"])
    markers=read_tsv(paths["markers"])
    resistance=read_tsv(paths["resistance"])
    fws=read_tsv(paths["fws"])

    for df, name in [(samples,"samples"),(markers,"markers"),(resistance,"resistance"),(fws,"fws")]:
        sample_col=canonical_column(df,"Sample","sample")
        if sample_col is None:
            raise ValueError(f"{name}: missing Sample/sample column")
        df.rename(columns={sample_col:"Sample"}, inplace=True)
        print(f"{name:<12}: {len(df):>7,} rows x {len(df.columns):>3} columns; duplicate IDs={df['Sample'].duplicated().sum()}")

    # Use QC-pass metadata, matching the manuscript source population.
    require(samples,["QC pass","Population","Study","Country","Year"],"samples")
    samples_qc=samples[samples["QC pass"].eq(True)].copy()
    print(f"\nQC-pass metadata: {len(samples_qc):,}")

    # Inner one-to-one merge across the four study resources.
    merged=(samples_qc
            .merge(markers,on="Sample",how="inner",validate="one_to_one")
            .merge(resistance,on="Sample",how="inner",validate="one_to_one")
            .merge(fws,on="Sample",how="inner",validate="one_to_one"))
    print(f"One-to-one matched records: {len(merged):,}")

    sea=merged[merged["Population"].isin(["AS-SE-E","AS-SE-W"])].copy()
    print(f"AS-SE-E / AS-SE-W eligible: {len(sea):,}")

    if "Artemisinin" not in sea.columns:
        raise ValueError("Expected 'Artemisinin' column not found in inferred resistance table.")

    norm=sea["Artemisinin"].astype(str).str.strip().str.lower()
    mapping={"resistant":1,"sensitive":0}
    y=norm.map(mapping)
    n_und=int(y.isna().sum())
    final=sea.loc[y.notna()].copy()
    final["artemisinin_resistant"]=y.loc[y.notna()].astype(int).to_numpy()

    print(f"Undetermined/excluded: {n_und:,}")
    print(f"Final analytical cohort: {len(final):,}")

    # Assertions reproduce the manuscript cohort and protect against silent source drift.
    if len(sea)!=7768:
        print(f"[WARNING] Expected 7,768 eligible SEA samples, observed {len(sea):,}. Check source-version alignment.")
    if n_und!=698:
        print(f"[WARNING] Expected 698 undetermined samples, observed {n_und:,}. Check source-version alignment.")
    if len(final)!=7070:
        print(f"[WARNING] Expected final n=7,070, observed {len(final):,}. Check source-version alignment.")

    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    final.to_csv(out,sep="\t",index=False)
    print(f"[SAVED] {out}")

    print("\nOutcome:")
    print(final["artemisinin_resistant"].value_counts().sort_index().rename(index={0:"Sensitive",1:"Resistant"}))
    print("\nCountry:")
    print(final.groupby("Country").size().sort_values(ascending=False))
    print(f"\nStudies: {final['Study'].nunique()}")
    print("="*72)

if __name__=="__main__":
    main()
