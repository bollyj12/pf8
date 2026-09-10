#!/usr/bin/env python
"""Run the reproducibility workflow from raw tables through model outputs."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def run(cmd):
    print("\n>>>"," ".join(map(str,cmd)),flush=True)
    subprocess.run(cmd,check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--skip-download",action="store_true")
    ap.add_argument("--skip-filter",action="store_true")
    ap.add_argument("--quick",action="store_true",help="Run only Full XGBoost nested CV + Full LOCO + XGBoost SHAP after filtering.")
    args=ap.parse_args()
    py=sys.executable
    data="data/derived/pf8_sea_artemisinin_clean.tsv"
    if not args.skip_download:
        run([py,"scripts/download_raw_pf8.py"])
    if not args.skip_filter:
        run([py,"scripts/filter_raw_pf8.py"])
    run([py,"pf8_data_loader.py"])
    run([py,"scripts/preprocessing.py","--data",data,"--set","Full"])

    if args.quick:
        jobs=[
          [py,"scripts/nested_study_grouped_cv.py","--data",data,"--set","Full","--model","XGBoost","--outdir","results/nested_cv"],
          [py,"scripts/loco.py","--data",data,"--set","Full","--outdir","results/loco"],
          [py,"scripts/shap_analysis.py","--data",data,"--model","XGBoost","--outdir","results/shap"],
        ]
    else:
        jobs=[]
        for s,m in [("Full","ElasticNet"),("Full","RandomForest"),("Full","XGBoost"),
                    ("Strict","XGBoost"),("ExtendedStrict","XGBoost")]:
            jobs.append([py,"scripts/nested_study_grouped_cv.py","--data",data,"--set",s,"--model",m,"--outdir","results/nested_cv"])
        jobs += [
          [py,"scripts/loco.py","--data",data,"--set","Full","--outdir","results/loco"],
          [py,"scripts/loco.py","--data",data,"--set","ExtendedStrict","--outdir","results/loco"],
          [py,"scripts/shap_analysis.py","--data",data,"--model","RF","--outdir","results/shap"],
          [py,"scripts/shap_analysis.py","--data",data,"--model","XGBoost","--outdir","results/shap"],
        ]
    for j in jobs: run(j)
    run([py,"scripts/figures/generate_all_figures.py"])\n    print("\nPIPELINE COMPLETE.")

if __name__=="__main__":
    main()
