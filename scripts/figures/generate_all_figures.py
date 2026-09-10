#!/usr/bin/env python
"""Generate Figures 1-5 for the manuscript."""
from pathlib import Path
import subprocess, sys

HERE=Path(__file__).resolve().parent
scripts=[
    "figure1_cohort_flow.py",
    "figure1b_cohort_construction.py",
    "figure2_country_prevalence.py",
    "figure3_nested_model_discrimination.py",
    "figure4_loco_transportability.py",
    "figure5_shap_global_attribution.py",
]
for s in scripts:
    cmd=[sys.executable,str(HERE/s)]
    print("\n>>>"," ".join(cmd))
    subprocess.run(cmd,check=True)
print("\nAll manuscript figures generated in results/figures/.")
