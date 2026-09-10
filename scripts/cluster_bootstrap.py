import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

parser=argparse.ArgumentParser()
parser.add_argument("--pred-a", required=True, help="Prediction CSV with Study,y_true,prob columns")
parser.add_argument("--pred-b", help="Optional second prediction CSV for paired AUC difference")
parser.add_argument("--out", required=True)
parser.add_argument("--replicates", type=int, default=2000)
parser.add_argument("--seed", type=int, default=20260822)
args=parser.parse_args()

a=pd.read_csv(args.pred_a)
rng=np.random.default_rng(args.seed)
study_order=list(a["Study"].astype(str).drop_duplicates())
groups_a={s:g for s,g in a.assign(Study=a.Study.astype(str)).groupby("Study",sort=False)}

if args.pred_b:
    b=pd.read_csv(args.pred_b)
    b=b[["Sample","prob"]].rename(columns={"prob":"prob_b"})
    a=a.merge(b,on="Sample",validate="one_to_one")
    groups_a={s:g for s,g in a.assign(Study=a.Study.astype(str)).groupby("Study",sort=False)}

vals=[]
for _ in range(args.replicates):
    ids=rng.integers(0,len(study_order),size=len(study_order))
    g=pd.concat([groups_a[study_order[i]] for i in ids],ignore_index=True)
    if g.y_true.nunique()<2:
        continue
    auc_a=roc_auc_score(g.y_true,g.prob)
    vals.append(auc_a-roc_auc_score(g.y_true,g.prob_b) if args.pred_b else auc_a)

vals=np.asarray(vals)
result={"estimate":float(roc_auc_score(a.y_true,a.prob)),
        "ci_low":float(np.percentile(vals,2.5)),
        "ci_high":float(np.percentile(vals,97.5)),
        "valid_replicates":int(len(vals))}
if args.pred_b:
    result["estimate"]=float(roc_auc_score(a.y_true,a.prob)-roc_auc_score(a.y_true,a.prob_b))
    result["bootstrap_p"]=float(min(1,2*min(np.mean(vals<=0),np.mean(vals>=0))))
pd.DataFrame([result]).to_csv(args.out,index=False)
