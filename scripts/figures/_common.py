from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    out = FIGURES / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")
