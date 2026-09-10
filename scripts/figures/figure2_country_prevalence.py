#!/usr/bin/env python
"""Figure 2. Country-level prevalence of Pf8-inferred artemisinin partial resistance."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *

df=pd.read_csv(DATA/"derived"/"pf8_sea_artemisinin_clean.tsv",sep="\t",low_memory=False)
tab=(df.groupby("Country")["artemisinin_resistant"]
       .agg(["mean","count"])
       .sort_values("mean",ascending=False))
tab["pct"]=100*tab["mean"]

fig,ax=plt.subplots(figsize=(8,5))
bars=ax.bar(tab.index,tab["pct"])
ax.set_ylabel("Resistant class prevalence (%)")
ax.set_xlabel("Country")
ax.set_ylim(0,100)
ax.set_title("Country-level prevalence of Pf8-inferred artemisinin partial resistance")
for b,(country,row) in zip(bars,tab.iterrows()):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,
            f"{row['pct']:.1f}%\n(n={int(row['count']):,})",
            ha="center",va="bottom",fontsize=9)
fig.tight_layout()
save(fig,"Figure_2_country_prevalence.png")
