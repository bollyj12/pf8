#!/usr/bin/env python
"""Figure 4. Leave-one-country-out geographical transportability."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *

def locate(name):
    candidates=[RESULTS/name, RESULTS/"loco"/name]
    return next((p for p in candidates if p.exists()),None)

p1=locate("loco_tuned_Full_XGBoost.csv")
p2=locate("loco_tuned_ExtendedStrict_XGBoost.csv")
if not p1 or not p2:
    raise SystemExit("LOCO Full/ExtendedStrict CSV files not found.")

full=pd.read_csv(p1)
ext=pd.read_csv(p2)
m=full[["Country","ROC_AUC"]].merge(ext[["Country","ROC_AUC"]],on="Country",suffixes=("_Full","_ExtendedStrict"))
order=["Cambodia","Laos","Myanmar","Thailand","Vietnam"]
m["Country"]=pd.Categorical(m["Country"],categories=order,ordered=True)
m=m.sort_values("Country")

x=np.arange(len(m))
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(x,m["ROC_AUC_Full"],marker="o",label="Full XGBoost")
ax.plot(x,m["ROC_AUC_ExtendedStrict"],marker="o",label="Extended Strict XGBoost")
ax.set_xticks(x,m["Country"].astype(str))
ax.set_ylim(0.5,1.0)
ax.set_ylabel("ROC-AUC")
ax.set_xlabel("Held-out country")
ax.set_title("Leave-one-country-out geographical transportability")
ax.legend()
ax.grid(axis="y",alpha=0.25)
fig.tight_layout()
save(fig,"Figure_4_LOCO_transportability.png")
