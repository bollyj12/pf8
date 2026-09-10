import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, json, sys, time
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from scipy import sparse

SEED=42
import argparse
parser=argparse.ArgumentParser(description='Nested study-grouped CV for Pf8 artemisinin analysis')
parser.add_argument('--data', required=True, help='Path to pf8_sea_artemisinin_clean.tsv')
parser.add_argument('--set', dest='setname', choices=['Full','Strict','ExtendedStrict'], required=True)
parser.add_argument('--model', dest='modelname', choices=['ElasticNet','RandomForest','XGBoost'], required=True)
parser.add_argument('--outdir', required=True)
args=parser.parse_args()
setname=args.setname; modelname=args.modelname
OUT=Path(args.outdir); OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(args.data,sep='\t')
y=df.artemisinin_resistant.astype(int).to_numpy(); groups=df.Study.astype(str).to_numpy()
full=['Year','crt_72[C]','crt_74[M]','crt_75[N]','crt_76[K]','crt_72-76[CVMNK]','crt_93[T]','crt_97[H]','crt_218[I]','crt_220[A]','crt_271[Q]','crt_326[N]','crt_333[T]','crt_353[G]','crt_356[I]','crt_371[R]','dhfr_16[N]','dhfr_51[N]','dhfr_59[C]','dhfr_108[S]','dhfr_164[I]','dhfr_306[S]','dhps_436[S]','dhps_437[G]','dhps_540[K]','dhps_581[A]','dhps_613[A]','exo_415[E]','mdr1_86[N]','mdr1_184[Y]','mdr1_1034[S]','mdr1_1042[N]','mdr1_1226[F]','mdr1_1246[D]','arps10_127-128[VD]','fd_193[D]','mdr2_484[T]','mdr1_dup_call','pm2_dup_call','Fws']
strict=[c for c in full if c not in ['arps10_127-128[VD]','fd_193[D]','mdr2_484[T]']]
extstrict=[c for c in strict if c not in ['crt_326[N]','crt_356[I]']]
sets={'Full':full,'Strict':strict,'ExtendedStrict':extstrict}
cols=sets[setname]
# Outcome-blind one-hot schema is fixed once; numeric scaling is fitted separately per training split.
cat=[c for c in cols if c not in ['Year','Fws']]
D=pd.get_dummies(df[cat].astype(str),prefix=cat,prefix_sep='=',dtype=np.float32)
Xcat=sparse.csr_matrix(D.to_numpy(dtype=np.float32))
Xnum=df[[c for c in ['Year','Fws'] if c in cols]].to_numpy(dtype=np.float32)

def makeX(train_idx, test_idx):
    sc=StandardScaler(); trn=sc.fit_transform(Xnum[train_idx]); ten=sc.transform(Xnum[test_idx])
    return sparse.hstack([sparse.csr_matrix(trn),Xcat[train_idx]],format='csr'), sparse.hstack([sparse.csr_matrix(ten),Xcat[test_idx]],format='csr')

def configs(name):
 if name=='ElasticNet': return [
  {'C':0.1,'l1_ratio':0.25},{'C':1,'l1_ratio':0.5},{'C':10,'l1_ratio':0.75}]
 if name=='RandomForest': return [
  {'n_estimators':100,'max_depth':None,'min_samples_leaf':1,'max_features':'sqrt'},
  {'n_estimators':100,'max_depth':None,'min_samples_leaf':2,'max_features':'sqrt'},
  {'n_estimators':100,'max_depth':12,'min_samples_leaf':2,'max_features':'sqrt'},
  {'n_estimators':100,'max_depth':None,'min_samples_leaf':2,'max_features':0.5}]
 return [
  {'n_estimators':150,'max_depth':3,'learning_rate':.05,'subsample':.85,'colsample_bytree':.85,'min_child_weight':1,'reg_alpha':0,'reg_lambda':1},
  {'n_estimators':150,'max_depth':4,'learning_rate':.05,'subsample':.85,'colsample_bytree':.85,'min_child_weight':1,'reg_alpha':0,'reg_lambda':1},
  {'n_estimators':150,'max_depth':4,'learning_rate':.03,'subsample':.85,'colsample_bytree':.85,'min_child_weight':3,'reg_alpha':.1,'reg_lambda':3},
  {'n_estimators':150,'max_depth':3,'learning_rate':.1,'subsample':.7,'colsample_bytree':.85,'min_child_weight':3,'reg_alpha':.1,'reg_lambda':3}]

def est(name,cfg):
 if name=='ElasticNet': return LogisticRegression(solver='saga',max_iter=1200,tol=1e-3,random_state=SEED,**cfg)
 if name=='RandomForest': return RandomForestClassifier(class_weight='balanced',random_state=SEED,n_jobs=-1,**cfg)
 return XGBClassifier(objective='binary:logistic',eval_metric='logloss',random_state=SEED,n_jobs=-1,**cfg)

def metrics(yy,p):
 z=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(yy,z,labels=[0,1]).ravel()
 return {'ROC_AUC':roc_auc_score(yy,p),'PR_AUC':average_precision_score(yy,p),'MCC':matthews_corrcoef(yy,z),'Balanced_Accuracy':balanced_accuracy_score(yy,z),'Accuracy':accuracy_score(yy,z),'Precision':precision_score(yy,z,zero_division=0),'Sensitivity':recall_score(yy,z),'Specificity':tn/(tn+fp),'F1':f1_score(yy,z)}
outer=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
foldrows=[]; predrows=[]; parrows=[]
for fold,(tr,te) in enumerate(outer.split(np.zeros(len(y)),y,groups),1):
 inner=StratifiedGroupKFold(3,shuffle=True,random_state=100+fold)
 scores=[]
 for cfg in configs(modelname):
  ss=[]
  for itr,iva in inner.split(np.zeros(len(tr)),y[tr],groups[tr]):
   a=tr[itr]; b=tr[iva]; Xa,Xb=makeX(a,b)
   md=est(modelname,cfg); md.fit(Xa,y[a]); ss.append(roc_auc_score(y[b],md.predict_proba(Xb)[:,1]))
  scores.append(np.mean(ss))
 bi=int(np.argmax(scores)); best=configs(modelname)[bi]
 Xtr,Xte=makeX(tr,te); md=est(modelname,best); md.fit(Xtr,y[tr]); p=md.predict_proba(Xte)[:,1]
 mm=metrics(y[te],p); mm.update(Set=setname,Model=modelname,Fold=fold,N_test=len(te),Inner_best_AUC=scores[bi]); foldrows.append(mm)
 parrows.append({'Set':setname,'Model':modelname,'Fold':fold,'Inner_best_AUC':scores[bi],'Best_params':json.dumps(best,sort_keys=True)})
 for i,pr in zip(te,p): predrows.append({'Row':int(i),'Sample':df.iloc[i].Sample,'Study':df.iloc[i].Study,'Country':df.iloc[i].Country,'Set':setname,'Model':modelname,'Fold':fold,'y_true':int(y[i]),'prob':float(pr),'pred':int(pr>=.5)})
 print(fold,round(mm['ROC_AUC'],4),round(scores[bi],4),best,flush=True)
pred=pd.DataFrame(predrows); fold=pd.DataFrame(foldrows); pars=pd.DataFrame(parrows)
pred.to_csv(OUT/f'nested_{setname}_{modelname}_pred.csv',index=False); fold.to_csv(OUT/f'nested_{setname}_{modelname}_fold.csv',index=False); pars.to_csv(OUT/f'nested_{setname}_{modelname}_params.csv',index=False)
mm=metrics(pred.y_true.to_numpy(),pred.prob.to_numpy()); mm.update(Set=setname,Model=modelname,N=len(pred)); pd.DataFrame([mm]).to_csv(OUT/f'nested_{setname}_{modelname}_summary.csv',index=False)
print('SUMMARY',mm,flush=True)
