#!/usr/bin/env python
"""Complete reproducibility pipeline for the Pf8 artemisinin-resistance workflow."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def resolve_python() -> str:
    candidates = [
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"\n>>> {' '.join(map(str, cmd))}", flush=True)
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True)


class Pf8Pipeline:
    def __init__(self, *, skip_download: bool, skip_filter: bool, quick: bool) -> None:
        self.py = resolve_python()
        self.data = "data/derived/pf8_sea_artemisinin_clean.tsv"
        self.skip_download = skip_download
        self.skip_filter = skip_filter
        self.quick = quick

    def download(self) -> None:
        run([self.py, "scripts/download_raw_pf8.py"])

    def filter_raw(self) -> None:
        run([self.py, "scripts/filter_raw_pf8.py"])

    def load(self) -> None:
        run([self.py, "pf8_data_loader.py"])

    def preprocess(self) -> None:
        run([self.py, "scripts/preprocessing.py", "--data", self.data, "--set", "Full"])

    def nested_models(self) -> None:
        if self.quick:
            jobs = [
                [self.py, "scripts/nested_study_grouped_cv.py", "--data", self.data, "--set", "Full", "--model", "XGBoost", "--outdir", "results/nested_cv"],
            ]
        else:
            jobs = []
            for s, m in [
                ("Full", "ElasticNet"),
                ("Full", "RandomForest"),
                ("Full", "XGBoost"),
                ("Strict", "XGBoost"),
                ("ExtendedStrict", "XGBoost"),
            ]:
                jobs.append([
                    self.py, "scripts/nested_study_grouped_cv.py",
                    "--data", self.data,
                    "--set", s,
                    "--model", m,
                    "--outdir", "results/nested_cv",
                ])
        for job in jobs:
            run(job)

    def loco(self) -> None:
        if self.quick:
            jobs = [
                [self.py, "scripts/loco.py", "--data", self.data, "--set", "Full", "--outdir", "results/loco"],
            ]
        else:
            jobs = [
                [self.py, "scripts/loco.py", "--data", self.data, "--set", "Full", "--outdir", "results/loco"],
                [self.py, "scripts/loco.py", "--data", self.data, "--set", "ExtendedStrict", "--outdir", "results/loco"],
            ]
        for job in jobs:
            run(job)

    def shap(self) -> None:
        if self.quick:
            jobs = [
                [self.py, "scripts/shap_analysis.py", "--data", self.data, "--model", "XGBoost", "--outdir", "results/shap"],
            ]
        else:
            jobs = [
                [self.py, "scripts/shap_analysis.py", "--data", self.data, "--model", "RF", "--outdir", "results/shap"],
                [self.py, "scripts/shap_analysis.py", "--data", self.data, "--model", "XGBoost", "--outdir", "results/shap"],
            ]
        for job in jobs:
            run(job)

    def figures(self) -> None:
        run([self.py, "scripts/figures/generate_all_figures.py"])

    def run(self) -> None:
        if not self.skip_download:
            self.download()
        if not self.skip_filter:
            self.filter_raw()
        self.load()
        self.preprocess()
        self.nested_models()
        self.loco()
        self.shap()
        self.figures()
        print("\nPIPELINE COMPLETE.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the complete Pf8 analytical pipeline.")
    ap.add_argument("--skip-download", action="store_true", help="Skip raw download step.")
    ap.add_argument("--skip-filter", action="store_true", help="Skip cohort reconstruction step.")
    ap.add_argument("--quick", action="store_true", help="Run a compact workflow that is useful for testing and demos.")
    args = ap.parse_args()

    pipeline = Pf8Pipeline(
        skip_download=args.skip_download,
        skip_filter=args.skip_filter,
        quick=args.quick,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
