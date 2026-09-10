#!/usr/bin/env python
"""Leakage-controlled preprocessing utilities for the Pf8 modelling scripts."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

FULL = [
'Year','crt_72[C]','crt_74[M]','crt_75[N]','crt_76[K]','crt_72-76[CVMNK]',
'crt_93[T]','crt_97[H]','crt_218[I]','crt_220[A]','crt_271[Q]','crt_326[N]',
'crt_333[T]','crt_353[G]','crt_356[I]','crt_371[R]','dhfr_16[N]','dhfr_51[N]',
'dhfr_59[C]','dhfr_108[S]','dhfr_164[I]','dhfr_306[S]','dhps_436[S]','dhps_437[G]',
'dhps_540[K]','dhps_581[A]','dhps_613[A]','exo_415[E]','mdr1_86[N]','mdr1_184[Y]',
'mdr1_1034[S]','mdr1_1042[N]','mdr1_1226[F]','mdr1_1246[D]','arps10_127-128[VD]',
'fd_193[D]','mdr2_484[T]','mdr1_dup_call','pm2_dup_call','Fws'
]
STRICT=[c for c in FULL if c not in ['arps10_127-128[VD]','fd_193[D]','mdr2_484[T]']]
EXTENDED_STRICT=[c for c in STRICT if c not in ['crt_326[N]','crt_356[I]']]
SETS={"Full":FULL,"Strict":STRICT,"ExtendedStrict":EXTENDED_STRICT}

FORBIDDEN_PREDICTORS={
    "Sample","Study","Country","Population","Country latitude","Country longitude",
    "Admin level 1 latitude","Admin level 1 longitude","kelch13_349-726_ns_changes",
    "Artemisinin","artemisinin_resistant"
}

class FoldPreprocessor:
    """Outcome-blind category schema + train-fold numeric standardization.

    The category dictionary is built from finite curated marker states without using y.
    Year/Fws StandardScaler is fitted ONLY to training indices.
    """
    def __init__(self, df:pd.DataFrame, setname:str):
        if setname not in SETS: raise ValueError(setname)
        self.cols=SETS[setname]
        absent=[c for c in self.cols if c not in df.columns]
        if absent: raise ValueError(f"Missing predictors: {absent}")
        leaked=set(self.cols)&FORBIDDEN_PREDICTORS
        if leaked: raise ValueError(f"Forbidden/leakage predictors in set: {sorted(leaked)}")
        self.cat=[c for c in self.cols if c not in ["Year","Fws"]]
        self.num=[c for c in ["Year","Fws"] if c in self.cols]
        dummies=pd.get_dummies(df[self.cat].astype(str),prefix=self.cat,prefix_sep="=",dtype=np.float32)
        self.cat_names=list(dummies.columns)
        self.Xcat=sparse.csr_matrix(dummies.to_numpy(np.float32))
        self.Xnum=df[self.num].to_numpy(np.float32)

    def split(self,train_idx,test_idx):
        scaler=StandardScaler()
        tr=scaler.fit_transform(self.Xnum[train_idx])
        te=scaler.transform(self.Xnum[test_idx])
        return (
          sparse.hstack([sparse.csr_matrix(tr),self.Xcat[train_idx]],format="csr"),
          sparse.hstack([sparse.csr_matrix(te),self.Xcat[test_idx]],format="csr")
        )

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/derived/pf8_sea_artemisinin_clean.tsv")
    ap.add_argument("--set",default="Full",choices=list(SETS))
    args=ap.parse_args()
    df=pd.read_csv(args.data,sep="\t")
    pp=FoldPreprocessor(df,args.set)
    print(f"{args.set}: {len(pp.cols)} source predictors")
    print(f"Numeric: {pp.num}")
    print(f"Categorical: {len(pp.cat)}")
    print(f"Encoded categorical columns: {len(pp.cat_names)}")
    print("Leakage controls: kelch13, country, population, Study, coordinates and outcome are not predictors.")
