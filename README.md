# Week 9 — Select the Winning Model and Final Test Evaluation

## What is included in merge

- `Src/week9_src_select_and_final_test.py` — nine-model comparison, selection lock, bias–variance diagnostic, and final one-time test evaluation.
- `Notebooks/week9_notebook_select_winner_final_test.ipynb` — same workflow explained in an organized Jupyter notebook.
- `Reports/week9/week9_report_select_winning_model.docx` — formatted written report.
- `Reports/week9/week9_reference_metrics/week9_verified_prior_week6_to_week8_metrics.csv` — audited historical Week 6–8 scores; Ridge values shown to the original Week 6 notebook's three-decimal precision.
- `Reports/week9/week9_reference_metrics/week9_figure_*.png` — pre-run illustrations generated from prior real validation results, **not** test results. Actual run exports charts under `Reports/week9/week9_figures/`.
- `Models/week9/README.md` — final saved-model pointer and expected manifest.
- `tests/test_week9_smoke.py` — isolated synthetic-only integrity test for the model-selection lock, validation gate, and test idempotence.
- `requirements_week9_additions.txt` — additional dependencies for notebook export. 
- `Reports/week9/week9_report_select_winning_model.docx` — formatted written report.
- `Reports/week9/week9_report_select_winning_model.pdf` — rendered and visually checked PDF reference copy of report.
- `Reports/week9/week9_notebook_select_winner_final_test.pdf` — executed notebook PDF supplied from local run.
- `Reports/week9/week9_tables/` — nine-model comparison, ranking, selection lock, final-test summary, 3-partition metrics and 21-month test error CSV.
- `Reports/week9/week9_figures/` — all four original Week 9 figures from the actual run.
- `Reports/week9/week9_reference_metrics/` — reference metrics and pre-run graphics required by the existing Week 9 source; retain these in the repository.
- `Reports/week9/week9_verified_results_for_report.md` — reconciled metrics, interpretation, provenance, and outstanding checks.
- `Reports/week9/week9_discussion_initial_post.md` — post-run discussion draft with the validated final-test result.
- `Reports/week9/week9_submission_checklist.md` — remaining local verification and submission items.

## Modeling and evaluation design

- **Candidates:** exactly nine previously trained configurations: Ridge_A/B/C, HGB_A/B/C, MLP_A/B/C.
- **Winner rule:** lowest **validation MAE**, then lowest validation RMSE, then predeclared complexity ordering for exact ties.
- **Final validation winner:** MLP_B (hidden layers 64,32), historical validation MAE 1,143.20 passengers.
- **Test policy:** select and save the winner first; verify the saved model reproduces Week 8 validation MAE and RMSE; only then predict the 264,291 previously untouched test observations.
- **No retuning after test:** one final evaluation of the saved, prior-training-fitted model. The source reuses saved results on repeated notebook runs.
- **Bias–variance diagnostic:** training/validation MSE for all nine configurations, with x-axis ordered by nominal capacity *within* families; cross-family complexity order is an approximation, not a formal global ranking. The gap is not a formal bias–variance decomposition.
- **Train/validation/test partitions:** previously established 1,062,603 / 239,395 / 264,291 eligible rows; all numerical preprocessing remains training-fitted.

## Verified results

- **Winner:** Week 8 `MLP_B`, hidden layers `(64, 32)`, selected by minimum validation MAE from nine prior configurations.
- **Validation MAE:** 1,143.20 passengers.
- **Final held-out test:** 264,291 rows, 21 forecast months from September 2024 through May 2026.
- **Test MAE:** 1,122.50 passengers; **Test RMSE:** 1,935.24 passengers; **Test R²:** 0.9390.
- **Train/validation/test MAE:** 1,161.63 / 1,143.20 / 1,122.50 passengers.
- **Model integrity:** selection lock, validation-reproduction gate, and test-no-refit flags are verified in the supplied result JSONs. Test evaluation need not be repeated.
- **Limitations:** model quality is promising in retrospective data but requires an operational acceptance threshold and publication-lag verification before deployment.

## VS Code / PowerShell workflow

1. Check the prerequisites. All should print `True`:

   ```powershell
   Test-Path .\Src\week5_src_features_dimensionality.py
   Test-Path .\Src\week6_src_ridge_models.py
   Test-Path .\Src\week8_src_mlp_neural_network.py
   Test-Path .\Src\week9_src_select_and_final_test.py
   Test-Path .\Reports\week6\week6_tables\week6_models_metrics_comparison.csv
   Test-Path .\Reports\week7\week7_tables\week7_three_model_comparison.csv
   Test-Path .\Reports\week8\week8_tables\week8_three_model_comparison.csv
   Test-Path .\Models\week8\week8_selected_mlp_neural_network.joblib
   ```

2. Use the existing Week 8 Python environment. Install only if necessary, without changing the environment used to serialize Week 8 MLP_B:

   ```powershell
   python -m pip install -r .\requirements.txt
   python -m pip install -r .\requirements_week9_additions.txt
   python -c "import pandas,numpy,sklearn,matplotlib,joblib,nbformat; print('Week 9 dependencies OK')"
   python -m py_compile .\Src\week9_src_select_and_final_test.py
   ```

   If the saved joblib model cannot be loaded because of a scikit-learn version mismatch, restore the **same compatible environment used in Week 8** rather than silently upgrading model code or retraining.

3. Optional code-integrity test:

   ```powershell
   python .\tests\test_week9_smoke.py
   ```

   It uses artificial data in a temporary directory and will not overwrite research files.

4. **Recommended safety check before final test:** run the validation-only comparison (does not access test), and confirm MLP_B and nine rows:

   ```powershell
   python .\Src\week9_src_select_and_final_test.py --validation-only
   Import-Csv .\Reports\week9\week9_tables\week9_nine_model_comparison.csv | Format-Table -AutoSize
   Get-Content .\Reports\week9\week9_tables\week9_selection_lock.json
   ```

   The original Week 6–8 CSV files are preferred. If one is absent, program uses its included historical reference. If a local source conflicts materially with reference, script stops and won't combining inconsistent metrics**. 

5. **Run the Week 9 notebook:** open `Notebooks/week9_notebook_select_winner_final_test.ipynb`, select the same Python kernel used in Week 8, click **Run All**.

6. Inspect the actual results:

   ```powershell
   Import-Csv .\Reports\week9\week9_tables\week9_nine_model_comparison.csv | Format-Table -AutoSize
   Import-Csv .\Reports\week9\week9_tables\week9_winner_train_validation_test_metrics.csv | Format-Table -AutoSize
   Get-Content .\Reports\week9\week9_tables\week9_test_winner_summary.json
   Get-Content .\Reports\week9\week9_verified_results_for_report.md
   Get-ChildItem .\Reports\week9\week9_figures -File
   Get-ChildItem .\Models\week9 -File
   ```

   After execution output folders contain nine-model comparison and validation ranking CSVs, selection lock JSON, final winning-model train/validation/test table, test summary JSON, test monthly MAE, four figures, and the final model-pointer manifest.

Repository: https://github.com/Laoshi88/ADAN8888-Airline-Demand-Forecasting