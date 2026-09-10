import argparse, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

SEED=42
parser=argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--model", choices=["RF","XGBoost"], required=True)
parser.add_argument("--outdir", required=True)
args=parser.parse_args()
OUT=Path(args.outdir); OUT.mkdir(parents=True,exist_ok=True)

df=pd.read_csv(args.data,sep="\t")
y=df["artemisinin_resistant"].astype(int).to_numpy()
cols=['Year','crt_72[C]','crt_74[M]','crt_75[N]','crt_76[K]','crt_72-76[CVMNK]',
'crt_93[T]','crt_97[H]','crt_218[I]','crt_220[A]','crt_271[Q]','crt_326[N]',
'crt_333[T]','crt_353[G]','crt_356[I]','crt_371[R]','dhfr_16[N]','dhfr_51[N]',
'dhfr_59[C]','dhfr_108[S]','dhfr_164[I]','dhfr_306[S]','dhps_436[S]','dhps_437[G]',
'dhps_540[K]','dhps_581[A]','dhps_613[A]','exo_415[E]','mdr1_86[N]','mdr1_184[Y]',
'mdr1_1034[S]','mdr1_1042[N]','mdr1_1226[F]','mdr1_1246[D]','arps10_127-128[VD]',
'fd_193[D]','mdr2_484[T]','mdr1_dup_call','pm2_dup_call','Fws']
cat=[c for c in cols if c not in ["Year","Fws"]]
D=pd.get_dummies(df[cat].astype(str),prefix=cat,prefix_sep="=",dtype=np.float32)
cat_names=list(D.columns)
source=["Year","Fws"]+[next(c for c in cat if n.startswith(c+"=")) for n in cat_names]
names=["Year","Fws"]+cat_names
sc=StandardScaler()
X=np.hstack([sc.fit_transform(df[["Year","Fws"]].to_numpy(np.float32)),D.to_numpy(np.float32)])

if args.model=="RF":
    cfg={'n_estimators':100,'max_depth':None,'min_samples_leaf':2,'max_features':'sqrt'}
    md=RandomForestClassifier(class_weight="balanced",random_state=SEED,n_jobs=-1,**cfg)
else:
    cfg={'n_estimators':150,'max_depth':4,'learning_rate':.05,'subsample':.85,
         'colsample_bytree':.85,'min_child_weight':1,'reg_alpha':0,'reg_lambda':1}
    md=XGBClassifier(objective="binary:logistic",eval_metric="logloss",
                     random_state=SEED,n_jobs=-1,**cfg)

md.fit(X,y)
rng=np.random.default_rng(SEED)
idx=rng.choice(len(y),1500,replace=False)
Xs=X[idx]
ex=shap.TreeExplainer(md)
try:
    sv=ex.shap_values(Xs,check_additivity=False,approximate=(args.model=="RF"))
except TypeError:
    sv=ex.shap_values(Xs,check_additivity=False)
if isinstance(sv,list): sv=sv[1]
sv=np.asarray(sv)
if sv.ndim==3: sv=sv[:,:,1]
imp=np.abs(sv).mean(axis=0)
enc=pd.DataFrame({"Encoded_feature":names,"Source_variable":source,"Mean_abs_SHAP":imp})
src=enc.groupby("Source_variable",as_index=False).Mean_abs_SHAP.sum().sort_values("Mean_abs_SHAP",ascending=False)
enc.to_csv(OUT/f"{args.model}_SHAP_encoded.csv",index=False)
src.to_csv(OUT/f"{args.model}_SHAP_source.csv",index=False)
(OUT/f"{args.model}_final_params.json").write_text(json.dumps(cfg,indent=2))
