# Pf8 full VS Code reproducibility project

This project contains the complete analysis workflow used for the manuscript:

1. official Pf8 raw-tabular data acquisition;
2. raw-table merge and Southeast Asian cohort filtering;
3. leakage-controlled preprocessing;
4. nested Study-grouped cross-validation;
5. study-cluster bootstrap;
6. leave-one-country-out (LOCO) validation;
7. SHAP global attribution.

## Official source

MalariaGEN Pf8, Zenodo DOI **10.5281/zenodo.18681980**, CC BY 4.0.

The four tabular source files used by this analysis are:

- `Pf8_samples.txt`
- `Pf8_fws.tsv`
- `Pf8_drug_resistance_marker_genotypes.tsv`
- `Pf8_inferred_resistance_status_classification.tsv`

The package bundles `Pf8_samples.txt` and the already-derived 7,070-sample analytical table.
Run `python scripts/download_raw_pf8.py` to retrieve and MD5-verify the remaining official raw tables directly from Zenodo.

**Important:** the full Pf8 release is much larger (Zenodo lists 22.1 GB and separate sequence/VCF resources). The manuscript does not use the 4.4-GB distance matrices, 13.2-GB diagnostic plot archive, CRAMs or full VCFs for this curated-marker analysis, so those are not duplicated into this VS Code project.

## Windows / VS Code

```cmd
setup_windows.bat
python scripts\download_raw_pf8.py
python scripts\filter_raw_pf8.py
python pf8_data_loader.py
python run_all.py --skip-download --skip-filter
```

For a shorter demonstration:

```cmd
python run_all.py --skip-download --skip-filter --quick
```

## Expected filtering checkpoints

- QC-pass matched Pf8 resource: 24,409
- AS-SE-E / AS-SE-W before outcome filtering: 7,768
- Undetermined artemisinin status excluded: 698
- Final cohort: 7,070
- Resistant: 3,903
- Sensitive: 3,167
- Studies: 23
- Countries: Cambodia, Laos, Myanmar, Thailand, Vietnam

## Leakage controls

- `kelch13_349-726_ns_changes` is never a predictor because Pf8 artemisinin status is Kelch13-derived.
- Country, Population, Study and coordinates are not predictors.
- Study is used only as the grouping variable in nested CV.
- One-hot marker categories are outcome-blind.
- Year and Fws scaling is fit separately inside each training split.
- Threshold-dependent metrics use a fixed 0.50 threshold.
- LOCO holds out an entire country for geographic transportability analysis.

## Script map

- `pf8_data_loader.py`: execution/status interface.
- `scripts/download_raw_pf8.py`: raw-data acquisition and MD5 verification.
- `scripts/filter_raw_pf8.py`: merge/filter raw Pf8 tables.
- `scripts/preprocessing.py`: predictor definitions and leakage-controlled fold transforms.
- `scripts/nested_study_grouped_cv.py`: nested 5x3 Study-grouped CV.
- `scripts/cluster_bootstrap.py`: study-cluster bootstrap uncertainty.
- `scripts/loco.py`: leave-one-country-out XGBoost.
- `scripts/shap_analysis.py`: TreeExplainer source-level SHAP.
- `run_all.py`: orchestrates the workflow.

## Reproducibility status

The modelling scripts are the portable scripts from the previous reproducibility package. This package makes the data acquisition, filtering and preprocessing stages explicit and runnable. The bundled derived table lets the modelling scripts be run immediately even before re-downloading the raw source tables.


## Figure reproduction

The package now includes scripts for all five manuscript figures:

- `scripts/figures/figure1_cohort_flow.py`
- `scripts/figures/figure2_country_prevalence.py`
- `scripts/figures/figure3_nested_model_discrimination.py`
- `scripts/figures/figure4_loco_transportability.py`
- `scripts/figures/figure5_shap_global_attribution.py`
- `scripts/figures/generate_all_figures.py`

Generate all figures with:

```cmd
python scripts\figures\generate_all_figures.py
```

Outputs are written at 300 dpi to:

`results\figures\`

Figure 1 recomputes cohort-flow counts from the four raw Pf8 tables when they
are present. Figures 2-5 use the derived analytical cohort and reproducibility
result tables. This ensures that the plotted values come from the same data
and model outputs reported in the manuscript.


## Generate manuscript figures and tables

After the analysis outputs are available, run:

```cmd
python visualize_pf8_results.py
```

This creates the `figures_tables/` directory containing:

- data structure overview
- country sample-size figure
- country resistance-prevalence figure
- nested-CV ROC-AUC confidence-interval figure
- nested model-performance table
- LOCO ROC-AUC and PR-AUC figures
- LOCO performance table
- XGBoost and Random Forest SHAP figures
- cohort summary table

Custom paths can be supplied with:

```cmd
python visualize_pf8_results.py --data data\derived\pf8_sea_artemisinin_clean.tsv --results results --outdir figures_tables
```
