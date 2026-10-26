
# Week 7 — Second Modeling Approach: Histogram Gradient Boosting Regression

## What is included in merge

- `Src/week7_src_hist_gradient_boosting.py` — Week 7 model training, hyperparameter evaluation, and validation source.
- `Notebooks/week7_notebook_model_training_tuning.ipynb` — same modeling pipeline in an explained Jupyter workflow.
- `Reports/week7/week7_report_second_modeling_approach.docx` — formatted written report.
- `tests/test_week7_smoke.py` — synthetic-data test for validating the Week 7 code without using actual BTS data.
- `Models/week7/README.md` — documentation for the saved winning model and model manifest.

## Modeling design

- **Method**: Histogram Gradient Boosting Regression, a nonlinear ensemble of decision trees using Week 5 `PC1`/`PC2` predictors.
- **Hyperparameter variants**:
  - `HGB_A` — learning_rate = 0.05, max_iter = 100, max_leaf_nodes = 15.
  - `HGB_B` — learning_rate = 0.05, max_iter = 250, max_leaf_nodes = 15.
  - `HGB_C` — learning_rate = 0.05, max_iter = 250, max_leaf_nodes = 31.
- **Fixed hyperparameters**: min_samples_leaf = 50, l2_regularization = 1.0, random_state = 42, and early_stopping = False.
- **Metrics**: training and validation MAE, RMSE, and R² on the original passenger-count scale, with a 0-passenger prediction floor.
- **Winner criterion**: *lowest validation MAE*, then validation RMSE, followed by the simpler leaf/iteration configuration for exact ties.
- **Model complexity**: HGB_A versus HGB_B evaluates additional boosting iterations; HGB_B versus HGB_C evaluates greater tree complexity.
- **Test set**: not scored, searched, or used for fitting or model selection; reserved for later final evaluation.
- **Full data**: Week 5 small samples are **not** used. Week 7 reconstructs the full Week 5 PCA transformations through the Week 6 source and original `Data/raw` BTS archives, preserving the Week 3 chronological 70%/15%/15% forecast-month split.
- **Week 6 benchmark**: Ridge_A (α = 1), with validation MAE of 3,257.18 passengers, provides a reference for comparing the new modeling approach.

## Step-by-step VS Code / PowerShell workflow

1. Check the prerequisites. All five checks should be `True`:

   ```powershell
   Test-Path .\Src\week2_src_t100_ingest_explore.py
   Test-Path .\Src\week3_src_eda_split.py
   Test-Path .\Src\week5_src_features_dimensionality.py
   Test-Path .\Src\week6_src_ridge_models.py
   Test-Path .\Src\week7_src_hist_gradient_boosting.py
   ```

   The official annual T-100 ZIP files must still exist under `Data/raw/`.

2. Install/check dependencies using the existing repository requirements:

   ```powershell
   python -m pip install -r .\requirements.txt
   python -c "import pandas, numpy, sklearn, matplotlib, joblib, threadpoolctl, nbformat; print('Week 7 dependencies OK')"
   python -m py_compile .\Src\week7_src_hist_gradient_boosting.py
   ```

   The source implementation requires pandas, NumPy, scikit-learn, matplotlib, joblib, and threadpoolctl.

   To run the optional synthetic-data code test:

   ```powershell
   python .\tests\test_week7_smoke.py
   ```

   This test verifies functionality only. Its results must not be included in the academic report.

3. Open `Notebooks/week7_notebook_model_training_tuning.ipynb`, select your previous Python kernel, click **Run All**, and wait until the final output says `All Week 7 checks passed.` and `===== WEEK 7 END =====`.

   - To run only from the command line, use `python .\Src\week7_src_hist_gradient_boosting.py`.
   - You do not need to run both the notebook and standalone Python file because they execute the same modeling workflow.

4. Verify the new output folders:

   ```powershell
   Get-ChildItem .\Reports\week7\week7_tables -File
   Get-ChildItem .\Reports\week7\week7_figures -File
   Get-ChildItem .\Models\week7 -File
   Get-Content .\Reports\week7\week7_verified_results_for_report.md
   ```

   Review the three-model comparison:

   ```powershell
   Import-Csv .\Reports\week7\week7_tables\week7_three_model_comparison.csv |
   Format-Table -AutoSize
   ```

   Confirm that all three configurations have training and validation MAE, RMSE, and R² results.