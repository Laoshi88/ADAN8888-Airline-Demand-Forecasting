# Week 10 — Data-Centric AI: Training Error Analysis and MLP_B Refit

## What is included in merge

- `Src/week10_src_data_enhancement_error_analysis_refit.py` — reusable training-only error audit; calendar-exact lag generation; three incremental data improvements; validation-based selection lock; and final historical test reporting.
- `Notebooks/week10_notebook_data_centric_ai.ipynb` — same workflow explained in an organized Jupyter notebook.
- `Reports/week10/week10_report_data_centric_ai.docx` and `.pdf` — finalized empirical Week 10 report with all recorded BTS results, validation winner, 4 figures, and operational limitations.
- `Reports/week10/week10_verified_results_for_report.md` — documented evidence, metrics, and interpretation safeguards.
- `Reports/week10/week10_tables/` — all generated CSV and JSON metrics, rankings, missingness audit, and training residual data.
- `Reports/week10/week10_figures/` — all generated figures; final report uses a curated subset.
- `Reports/week10/week10_notebook_data_centric_ai.pdf` — executed notebook PDF with `All Week 10 checks passed.`
- `tests/test_week10_smoke.py` — independent **synthetic** end-to-end test; its numerical metrics must never be reported as real BTS results.
- `requirements_week10_additions.txt` — dependencies already used by earlier weeks; retain the environment compatible with the saved Week 8/9 estimator.
- `Models/week10/` — original selected-model bundle, manifest, and README.
- `Data/processed/week10/` — original 500-row enhanced data samples for each split; **not** the full model-training inputs.

## Modeling design and verified results

- **Architecture:** Week 9-selected MLP_B `(64, 32)`, same training hyperparameters, expanding input-layer dimensionality across enhancement stages.
- **Reference:** Original MLP_B uses 2 features (PC1, PC2); validation MAE 1,143.20; validation RMSE 1,988.78.
- **Improvement 1:** Exact-calendar demand lags, complete rolling mean, monthly/annual log changes, and missingness indicators. The 16-feature model achieved validation MAE **809.33**, a 29.20% reduction.
- **Improvement 2:** Adds operational and data-quality features; 30 inputs, validation MAE **819.80** (worse than temporal-only).
- **Improvement 3:** Adds seasonal and route context; 38 inputs, validation MAE **870.35** (worse again despite lower training error).
- **Winner:** `MLP_B_temporal` by validation MAE, then RMSE and fewer parameters for ties. It uses 16 inputs and 3,201 trainable parameters.
- **Selected model MAE:** training **808.17**, validation **809.33**, historical test **790.29** passengers. Test RMSE **1,283.12**, R² **0.9732**.
- **Test policy:** The Week 9 test period was examined previously; Week 10 test is a **reused retrospective period**, not a new independent holdout. The selection lock was written before Week 10 test scoring.
- **Operational limitation:** Official BTS monthly inputs may be published too late for real-time issuance, and no business error-acceptance threshold is supplied.

## Step-by-step VS Code / PowerShell workflow

1. **Check prerequisites.** Every statement should return `True`. The model file must be the exact Week 9-validated artifact; a newly fitted replacement is not acceptable.

   ```powershell
   Test-Path .\Src\week5_src_features_dimensionality.py
   Test-Path .\Src\week8_src_mlp_neural_network.py
   Test-Path .\Src\week10_src_data_enhancement_error_analysis_refit.py
   Test-Path .\Models\week8\week8_selected_mlp_neural_network.joblib
   Test-Path .\Models\week9\week9_final_model_manifest.json
   Test-Path .\Reports\week9\week9_tables\week9_selection_lock.json
   Test-Path .\Data\raw
   ```

   **If `False` appears, stop.** Restore the missing source/model/manifest from your previous weeks; do not retrain the Week 9 original simply to bypass this check.

2. **Verify the completed evidence:**

   ```powershell
   Test-Path .\Src\week10_src_data_enhancement_error_analysis_refit.py
   Test-Path .\Notebooks\week10_notebook_data_centric_ai.ipynb
   Test-Path .\Reports\week10\week10_tables\week10_incremental_ablation.csv
   Test-Path .\Reports\week10\week10_tables\week10_selection_lock.json
   Test-Path .\Reports\week10\week10_tables\week10_final_test_result.json
   Test-Path .\Models\week10\week10_final_model_manifest.json
   ```

   All should return `True`. Compare the selected model in the two JSON files; it must be `MLP_B_temporal`.

3. **Check the Python environment and source without changing scikit-learn versions unnecessarily.** Pick the same Python kernel used for the Week 9 saved model. Install only missing requirements using the existing repository requirements as appropriate.

   ```powershell
   python -c "import pandas,numpy,sklearn,matplotlib,joblib,threadpoolctl,nbformat; print('Week 10 dependencies OK')"
   python -m py_compile .\Src\week10_src_data_enhancement_error_analysis_refit.py
   python .\tests\test_week10_smoke.py
   ```

   The smoke test has **synthetic data only** and reports no real BTS metrics.

4. **Run the real full-data notebook.** Open `Notebooks/week10_notebook_data_centric_ai.ipynb`; select your established kernel and click **Run All**. This runs the source only once. Keep VS Code open until the final cell finishes. It should display:

   ```text
   All Week 10 checks passed.
   ===== WEEK 10 END =====
   ```

   For command-line execution **instead** of the notebook, use `python .\Src\week10_src_data_enhancement_error_analysis_refit.py`. Do **not** run both from scratch; the notebook can read completed cached outputs.

5. **Inspect the required results.** The following files are generated from the full local BTS dataset:

   ```powershell
   Import-Csv .\Reports\week10\week10_tables\week10_incremental_ablation.csv | Format-Table -AutoSize
   Import-Csv .\Reports\week10\week10_tables\week10_train_validation_test_metrics.csv | Format-Table -AutoSize
   Get-Content .\Reports\week10\week10_tables\week10_selection_lock.json
   Get-Content .\Reports\week10\week10_tables\week10_final_test_result.json
   Get-ChildItem .\Reports\week10\week10_figures -File
   Get-Content .\Reports\week10\week10_verified_results_for_report.md
   ```

## Interpretation safeguards

- Week 10 synthetic smoke test is separate from this official BTS run; **only the recorded BTS outputs** are used in the report.
- An enhancement is not necessarily beneficial; the additional quality and route/seasonality stages made validation MAE worse compared with temporal history alone.
- The winning temporal stage has a larger input layer and more trainable weights than the original model despite the same hidden-layer architecture.
- Training errors are diagnostic; production readiness requires as-of availability, business thresholds, and new prospective validation.

Repository: https://github.com/Laoshi88/ADAN8888-Airline-Demand-Forecasting
