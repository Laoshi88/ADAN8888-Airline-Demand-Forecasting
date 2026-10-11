# ADAN8888 Week 5 — Feature Engineering and Dimensionality Reduction

## Week 5 files

- `Src/week5_src_features_dimensionality.py`: reusable feature creation, chronological splitting through Week 3 logic, train-fitted PCA, compact output export.
- `Notebooks/week5_notebook_features_dimensionality.ipynb`: narrative and runnable notebook calling the same source.
- `Reports/week5/week5_report_engineer_features.docx`: formatted written report.
- `Reports/week5/week5_submission_checklist.md`: final artifact checks.
- `Data/processed/week5/README.md`: documented output location. Small processed samples are generated when code runs.

## Why this architecture

Earlier Week 2 and Week 3 `.py` sources must exist in the repository and the official annual BTS T-100 ZIPs must be in `Data/raw/`.

- The chronological Week 3 split is recreated through its own `prepare_model_data` and `chronological_split` methods, guaranteeing the same eligibility and forecast-month rounding.
- A unique carrier-route-month is the unit of analysis. The following calendar month's passengers are the target. PCA excludes the target.

**No synthetic augmentation** is performed.

## Steps: execute once in Powershell

1. Confirm the files exist:

   ```powershell
   Test-Path .\Src\week2_src_t100_ingest_explore.py
   Test-Path .\Src\week3_src_eda_split.py
   Test-Path .\Src\week5_src_features_dimensionality.py
   Test-Path .\Notebooks\week5_notebook_features_dimensionality.ipynb
   ```

2. Install/check required existing dependencies (no new requirements file is needed):

   ```powershell
   python -m pip install -r .\requirements.txt
   python -m py_compile .\Src\week5_src_features_dimensionality.py
   ```

3. Open `Notebooks/week5_notebook_features_dimensionality.ipynb`, select Week 4 Python kernel, **Run All**. The final cell must say `All Week 5 checks passed.` and `===== WEEK 5 END =====`. If a data/source file is unavailable, execution stops with a clear error; no results are fabricated.

4. Atternative: run standalone `.py` file, `python .\Src\week5_src_features_dimensionality.py`.

5. Inspect generated artifacts:

   ```powershell
   Get-ChildItem .\Reports\week5\week5_tables -File
   Get-ChildItem .\Reports\week5\week5_figures -File
   Get-ChildItem .\Data\processed\week5 -File
   Import-Csv .\Reports\week5\week5_tables\week5_component_summary.csv | Format-Table -AutoSize
   ```

## Methodological limitations

- PCA retains input variance, not necessarily predictive signal. Compare against the Week 4 representation in later modeling.
- High-volume markets, data gaps, and the COVID-era demand shock should not be erased by synthetic samples or automated outlier deletion.
- The feature inputs depend on completed month *t* aggregates being available before forecasting *t+1*. Official BTS reporting latency is an operational constraint for deployment.
- Large full-split CSV exports are deliberately omitted from GitHub; raw official archives and reproducible code remain the source of truth.
