#!/usr/bin/env python
"""Figure 3. Nested study-grouped model discrimination with cluster-bootstrap CIs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *

# Prefer the concise CI file copied with the reproducibility outputs.
candidates=[
    RESULTS/"nested_primary_model_CI_2000.csv",
    RESULTS/"nested_cv"/"nested_primary_model_CI_2000.csv",
]
p=next((x for x in candidates if x.exists()),None)
if p is None:
    raise SystemExit("nested_primary_model_CI_2000.csv not found in results/")

df=pd.read_csv(p)
print("Columns:",list(df.columns))

def col_like(words):
    for c in df.columns:
        low=c.lower()
        if all(w in low for w in words):
            return c
    return None

label_col=col_like(["model"]) or df.columns[0]
auc_col=col_like(["auc"]) or df.select_dtypes(include="number").columns[0]
lo_col=next((c for c in df.columns if "lower" in c.lower() or "lo"==c.lower()),None)
hi_col=next((c for c in df.columns if "upper" in c.lower() or "hi"==c.lower()),None)

if lo_col is None or hi_col is None:
    # Handle common ci_low/ci_high names
    lo_col=next((c for c in df.columns if "ci" in c.lower() and "low" in c.lower()),None)
    hi_col=next((c for c in df.columns if "ci" in c.lower() and "high" in c.lower()),None)
if lo_col is None or hi_col is None:
    raise SystemExit(f"Could not identify CI columns in {list(df.columns)}")

labels=df[label_col].astype(str).tolist()
auc=df[auc_col].astype(float).to_numpy()
lo=df[lo_col].astype(float).to_numpy()
hi=df[hi_col].astype(float).to_numpy()
y=np.arange(len(df))

fig,ax=plt.subplots(figsize=(8,5.5))
ax.errorbar(auc,y,xerr=[auc-lo,hi-auc],fmt="o",capsize=4)
ax.set_yticks(y,labels)
ax.set_xlabel("ROC-AUC")
ax.set_xlim(max(0,min(lo)-0.03),min(1,max(hi)+0.03))
ax.set_title("Nested study-grouped model discrimination")
ax.grid(axis="x",alpha=0.25)
fig.tight_layout()
save(fig,"Figure_3_nested_model_discrimination.png")
