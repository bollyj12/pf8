import argparse, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score, f1_score, recall_score, confusion_matrix)
from xgboost import XGBClassifier

SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--set", dest="setname", choices=["Full", "ExtendedStrict"], required=True)
parser.add_argument("--outdir", required=True)
args = parser.parse_args()

OUT = Path(args.outdir)
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.data, sep="\t")
y = df["artemisinin_resistant"].astype(int).to_numpy()
groups = df["Study"].astype(str).to_numpy()

full = ['Year','crt_72[C]','crt_74[M]','crt_75[N]','crt_76[K]','crt_72-76[CVMNK]',
'crt_93[T]','crt_97[H]','crt_218[I]','crt_220[A]','crt_271[Q]','crt_326[N]',
'crt_333[T]','crt_353[G]','crt_356[I]','crt_371[R]','dhfr_16[N]','dhfr_51[N]',
'dhfr_59[C]','dhfr_108[S]','dhfr_164[I]','dhfr_306[S]','dhps_436[S]','dhps_437[G]',
'dhps_540[K]','dhps_581[A]','dhps_613[A]','exo_415[E]','mdr1_86[N]','mdr1_184[Y]',
'mdr1_1034[S]','mdr1_1042[N]','mdr1_1226[F]','mdr1_1246[D]','arps10_127-128[VD]',
'fd_193[D]','mdr2_484[T]','mdr1_dup_call','pm2_dup_call','Fws']
strict = [c for c in full if c not in ['arps10_127-128[VD]','fd_193[D]','mdr2_484[T]']]
extended = [c for c in strict if c not in ['crt_326[N]','crt_356[I]']]
cols = full if args.setname == "Full" else extended

cat = [c for c in cols if c not in ["Year","Fws"]]
D = pd.get_dummies(df[cat].astype(str), prefix=cat, prefix_sep="=", dtype=np.float32)
Xcat = sparse.csr_matrix(D.to_numpy(np.float32))
Xnum = df[["Year","Fws"]].to_numpy(np.float32)

def transform(train_idx, test_idx):
    sc = StandardScaler()
    A = sc.fit_transform(Xnum[train_idx])
    B = sc.transform(Xnum[test_idx])
    return (sparse.hstack([A, Xcat[train_idx]], format="csr"),
            sparse.hstack([B, Xcat[test_idx]], format="csr"))

configs = [
 {'n_estimators':150,'max_depth':3,'learning_rate':.05,'subsample':.85,'colsample_bytree':.85,'min_child_weight':1,'reg_alpha':0,'reg_lambda':1},
 {'n_estimators':150,'max_depth':4,'learning_rate':.05,'subsample':.85,'colsample_bytree':.85,'min_child_weight':1,'reg_alpha':0,'reg_lambda':1},
 {'n_estimators':150,'max_depth':4,'learning_rate':.03,'subsample':.85,'colsample_bytree':.85,'min_child_weight':3,'reg_alpha':.1,'reg_lambda':3},
 {'n_estimators':150,'max_depth':3,'learning_rate':.1,'subsample':.7,'colsample_bytree':.85,'min_child_weight':3,'reg_alpha':.1,'reg_lambda':3}
]

def model(cfg):
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                         random_state=SEED, n_jobs=-1, **cfg)

def metrics(yy, p):
    z = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, z, labels=[0,1]).ravel()
    return {
      "ROC_AUC": roc_auc_score(yy,p),
      "PR_AUC": average_precision_score(yy,p),
      "MCC": matthews_corrcoef(yy,z),
      "Balanced_Accuracy": balanced_accuracy_score(yy,z),
      "F1": f1_score(yy,z),
      "Sensitivity": recall_score(yy,z),
      "Specificity": tn/(tn+fp)
    }

rows=[]
for country in ["Cambodia","Laos","Myanmar","Thailand","Vietnam"]:
    te=np.where(df["Country"].to_numpy()==country)[0]
    tr=np.where(df["Country"].to_numpy()!=country)[0]
    inner=StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
    scores=[]
    for cfg in configs:
        ss=[]
        for ia,ib in inner.split(np.zeros(len(tr)), y[tr], groups[tr]):
            a=tr[ia]; b=tr[ib]
            A,B=transform(a,b)
            md=model(cfg); md.fit(A,y[a])
            ss.append(roc_auc_score(y[b], md.predict_proba(B)[:,1]))
        scores.append(np.mean(ss))
    best=configs[int(np.argmax(scores))]
    A,B=transform(tr,te)
    md=model(best); md.fit(A,y[tr])
    p=md.predict_proba(B)[:,1]
    row=metrics(y[te],p)
    row.update(Set=args.setname, Country=country, N=len(te),
               Inner_best_AUC=max(scores), Best_params=json.dumps(best,sort_keys=True))
    rows.append(row)

pd.DataFrame(rows).to_csv(OUT/f"loco_tuned_{args.setname}_XGBoost.csv",index=False)
