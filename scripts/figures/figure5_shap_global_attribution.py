#!/usr/bin/env python
"""Figure 5. SHAP-based global source-variable attribution across tree models."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *

def locate(name):
    candidates=[RESULTS/name, RESULTS/"shap"/name]
    return next((p for p in candidates if p.exists()),None)

rfp=locate("RF_SHAP_source.csv")
xgp=locate("XGBoost_SHAP_source.csv")
if not rfp or not xgp:
    raise SystemExit("RF_SHAP_source.csv or XGBoost_SHAP_source.csv not found.")

rf=pd.read_csv(rfp)
xg=pd.read_csv(xgp)

def infer(df):
    text=[c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
    num=list(df.select_dtypes(include="number").columns)
    if not text or not num:
        raise ValueError(f"Could not infer SHAP columns from {list(df.columns)}")
    feature=text[0]
    value=next((c for c in num if "mean" in c.lower() or "abs" in c.lower() or "shap" in c.lower()), num[0])
    return feature, value

rf_f,rf_v=infer(rf)
xg_f,xg_v=infer(xg)
a=rf[[rf_f,rf_v]].rename(columns={rf_f:"feature",rf_v:"RF"})
b=xg[[xg_f,xg_v]].rename(columns={xg_f:"feature",xg_v:"XGB"})
m=a.merge(b,on="feature",how="outer").fillna(0)

# Convert each model's absolute SHAP to a share of its own total importance so the
# underlying relative contribution is interpretable for both models.
for c in ["RF","XGB"]:
    total=m[c].sum()
    if total > 0:
        m[c]=m[c]/total

m["combined"]=m[["RF","XGB"]].mean(axis=1)
m=m.sort_values("combined",ascending=False).head(12).sort_values("combined")
y=np.arange(len(m))

fig,ax=plt.subplots(figsize=(10,7))
bar_h=0.72
ax.barh(y-bar_h/2,m["RF"],height=bar_h,color="#4C72B0",alpha=0.85,label="Random Forest")
ax.barh(y+bar_h/2,m["XGB"],height=bar_h,color="#DD8452",alpha=0.8,label="XGBoost")

for i,row in enumerate(m.itertuples()):
    ax.text(row.RF + 0.008, y[i]-bar_h/2, f"{row.RF:.3f}", va="center", ha="left", fontsize=8, color="#2F4F75")
    ax.text(row.XGB + 0.008, y[i]+bar_h/2, f"{row.XGB:.3f}", va="center", ha="left", fontsize=8, color="#8C4B16")

ax.set_yticks(y)
ax.set_yticklabels(m["feature"], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Relative mean |SHAP| contribution within model")
ax.set_title("Top source variables driving model predictions\n(SHAP-based global attribution)")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.25)
ax.set_xlim(0, max(m[["RF","XGB"]].max().max() * 1.3, 0.15))
fig.tight_layout()
save(fig,"Figure_5_SHAP_global_attribution.png")
