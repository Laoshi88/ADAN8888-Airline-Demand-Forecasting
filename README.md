# Week 6 — First Modeling Approach: Ridge Regression

## What is included in merge

- `Src/week6_src_ridge_models.py` — reusable Week 6 training/evaluation source.
- `Notebooks/week6_notebook_ridge_model_training.ipynb` — same pipeline in an explained Jupyter workflow.
- `Reports/week6/week6_report_first_modeling_approach.docx` — professional, **four-page working report** aligned to the assignment. The results table and winner fields are deliberately blank because **actual Week 6 model outputs were not provided**. Do not submit until completing them.
- `Reports/week6/week6_discussion_initial_post.md` — assignment-aligned initial post; optional final-result sentence must be filled after execution if used.
- `Reports/week6/week6_submission_checklist.md` — submission gate.

## Modeling design

- **Method**: Ridge regression, a regularized linear regression on Week 5 `PC1`/`PC2`.
- **Hyperparameter variants**: alpha = 1, 10,000, and 1,000,000.
- **Metrics**: training and validation MAE, RMSE, and R² on passenger scale, with a 0-passenger prediction floor.
- **Winner criterion**: *lowest validation MAE*, then validation RMSE, then smaller alpha for exact ties.
- **Test set**: not scored, searched or used for fitting; saved for later final evaluation.
- **Full data**: Week 5 small samples are **not** used; Week 6 reconstructs the full Week 5 transformations from the original `Data/raw` annual BTS archives. This ensures the 70/15/15 chronological forecasting design carries through from Week 3.

## Step-by-step VS Code / PowerShell workflow

1. Check the prerequisites. All three checks should be `True`:
   ```powershell
   Test-Path .\Src\week2_src_t100_ingest_explore.py
   Test-Path .\Src\week3_src_eda_split.py
   Test-Path .\Src\week5_src_features_dimensionality.py
   ```
   The official annual T-100 ZIPs must still exist under `Data/raw/`.

2. Install/check dependencies using the existing repository requirements:
   ```powershell
   python -m pip install -r .\requirements.txt
   python -c "import pandas, numpy, sklearn, matplotlib, joblib, nbformat; print('Week 6 dependencies OK')"
   python -m py_compile .\Src\week6_src_ridge_models.py
   ```
   If `nbformat` is not installed, the VS Code Jupyter extension normally supplies notebook handling. The source implementation needs pandas, numpy, scikit-learn, matplotlib and joblib.

3. Open `Notebooks/week6_notebook_ridge_model_training.ipynb`, select your previous Python kernel, click **Run All** and wait until the last output says `All Week 6 checks passed.` and `===== WEEK 6 END =====`.
   - To run only from the command line, use `python .\Src\week6_src_ridge_models.py`.

4. Verify new output folders:
   ```powershell
   Get-ChildItem .\Reports\week6\week6_tables -File
   Get-ChildItem .\Reports\week6\week6_figures -File
   Get-ChildItem .\Models\week6 -File
   Get-Content .\Reports\week6\week6_verified_results_for_report.md
   ```
