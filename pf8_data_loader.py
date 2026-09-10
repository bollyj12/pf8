#!/usr/bin/env python
"""Console interface showing Pf8 loading and pipeline status."""
from pathlib import Path
import argparse, hashlib, pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw-dir",default="data/raw")
    ap.add_argument("--clean",default="data/derived/pf8_sea_artemisinin_clean.tsv")
    args=ap.parse_args()
    raw=Path(args.raw_dir)
    print("="*72)
    print("MalariaGEN Pf8 REPRODUCIBILITY INTERFACE")
    print("="*72)
    names=[
      "Pf8_samples.txt","Pf8_fws.tsv",
      "Pf8_drug_resistance_marker_genotypes.tsv",
      "Pf8_inferred_resistance_status_classification.tsv"
    ]
    print("\n[1] RAW SOURCE FILES")
    for n in names:
        p=raw/n
        print(f"  {'[OK]' if p.exists() else '[MISSING]'} {n}" + (f" ({p.stat().st_size/1024/1024:.2f} MB)" if p.exists() else ""))
    p=Path(args.clean)
    print("\n[2] DERIVED ANALYTICAL TABLE")
    if p.exists():
        df=pd.read_csv(p,sep="\t",low_memory=False)
        print(f"  [OK] {len(df):,} rows x {len(df.columns)} columns")
        print(f"  Studies: {df.Study.nunique()} | Countries: {df.Country.nunique()}")
        print(f"  Resistant: {(df.artemisinin_resistant==1).sum():,} | Sensitive: {(df.artemisinin_resistant==0).sum():,}")
        print("  Kelch13 is retained in the archival merged table but excluded from every predictor set.")
    else:
        print("  [MISSING] Run scripts/filter_raw_pf8.py")
    print("\n[3] PIPELINE")
    print("  raw Pf8 -> filter/merge -> leakage-controlled preprocessing")
    print("  -> nested study-grouped CV -> cluster bootstrap")
    print("  -> LOCO geographic transportability -> SHAP")
    print("="*72)

if __name__=="__main__":
    main()
