#!/usr/bin/env python
"""Download the four official Pf8 tabular source files from Zenodo.

Source:
  MalariaGEN Pf8, Zenodo DOI 10.5281/zenodo.18681980

The files are CC BY 4.0. Existing files are validated by MD5 unless --force is used.
"""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path
from urllib.request import urlopen, Request

BASE="https://zenodo.org/records/18681980/files/"
FILES={
    "Pf8_samples.txt":"c5832d7207a9ff2377f3d5c67e1c4582",
    "Pf8_fws.tsv":"50d555a152924d5e8b2f35087a50893c",
    "Pf8_drug_resistance_marker_genotypes.tsv":"c3de1d3d5b021af9e6d5a497e68dd47a",
    "Pf8_inferred_resistance_status_classification.tsv":"4a6062d66bc4472a7006b87dbe4763ca",
}
def md5(path,chunk=1024*1024):
    h=hashlib.md5()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def download(name,dest):
    url=BASE+name+"?download=1"
    req=Request(url,headers={"User-Agent":"Pf8-reproducibility/1.0"})
    print(f"[DOWNLOAD] {name}")
    with urlopen(req,timeout=120) as r, open(dest,'wb') as f:
        while True:
            chunk=r.read(1024*1024)
            if not chunk: break
            f.write(chunk)
    got=md5(dest)
    if got != FILES[name]:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {name}: {got}")
    print(f"[OK] {name} ({dest.stat().st_size/1024/1024:.2f} MB)")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",default="data/raw")
    ap.add_argument("--force",action="store_true")
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    for name,expected in FILES.items():
        p=out/name
        if p.exists() and not args.force:
            got=md5(p)
            if got==expected:
                print(f"[OK] already present and verified: {name}")
                continue
            print(f"[WARN] existing file failed MD5: {name}; downloading again.")
        try:
            download(name,p)
        except Exception as e:
            print(f"[ERROR] Could not download {name}: {e}")
            print("You can manually download it from DOI 10.5281/zenodo.18681980 and place it in data/raw/.")
            sys.exit(1)
    print("\nAll four Pf8 source tables are present and MD5 verified.")

if __name__=="__main__":
    main()
