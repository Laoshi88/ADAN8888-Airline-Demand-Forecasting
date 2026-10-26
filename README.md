# ADAN8888 Week 7 — Complete Execution and Submission Guide

**Use this as a merge package, not a new Git repository.** It replaces the two Week 7 drafts with corrected Week 7 versions; it does not delete prior weeks. Your existing root is:

```text
C:\Users\king-\Desktop\ADAN8888 - AA Project\Module 2\3. Assignment1\2. Project Folder - GitHub
```

## 1. Key changes versus the Week 7 draft

- Corrected an incomplete notebook cell (`results = ...`) that would prevent execution.
- Reduced the draft's eight-configuration grid to **three** well-defined complexity variations, as the assignment requests.
- Reused the existing Week 6 loader and Week 5 PCA to avoid a competing ingestion/preprocessing pipeline.
- Preserved the train-only fit, validation-only winner selection, untouched test set, and zero-floor policy from Week 6.
- Introduced a reusable, audited Python module; notebook narrative calls exactly the same functions.
- Added six (three configurations × two partitions) metric records, tabular comparison, model artifact, full-input validation diagnostics, plots and a compact output summary.
- Rebuilt the Word report into a professional formatted document with marked **real-results placeholders**, not invented metrics.

## 2. Package contents

```text
Src/week7_src_hist_gradient_boosting.py
Notebooks/week7_notebook_model_training_tuning.ipynb
Reports/week7/week7_report_second_modeling_approach.docx
Reports/week7/week7_discussion_initial_post.md
Reports/week7/week7_submission_checklist.md
Models/week7/README.md
tests/test_week7_smoke.py
README_WEEK7.md
```

When you run Week 7 on the actual BTS archives it creates:

```text
Reports/week7/week7_tables/
  week7_three_model_comparison.csv
  week7_metrics_long.csv
  week7_validation_ranking.csv
  week7_split_summary.csv
  week7_monthly_validation_mae.csv
  week7_winner_summary.json
Reports/week7/week7_figures/
  week7_figure_mae_by_variation.png
  week7_figure_rmse_by_variation.png
  week7_figure_validation_actual_vs_predicted.png
  week7_figure_monthly_validation_mae.png
Reports/week7/week7_verified_results_for_report.md
Models/week7/week7_selected_hist_gradient_boosting.joblib
Models/week7/week7_model_manifest.json
```

No synthetic test outputs are included in the actual submission package. The Week 7 model consumes the **full** earlier-week train and validation partitions; the prior Week 5 sample CSVs are not used.

## 3. PowerShell setup: run in VS Code

Open the existing project in VS Code. In the PowerShell terminal:

```powershell
cd "C:\Users\king-\Desktop\ADAN8888 - AA Project\Module 2\3. Assignment1\2. Project Folder - GitHub"
git rev-parse --show-toplevel
git status
```

Extract the ZIP's **contents** into that folder. Confirm these prerequisites:

```powershell
Test-Path .\Src\week2_src_t100_ingest_explore.py
Test-Path .\Src\week3_src_eda_split.py
Test-Path .\Src\week5_src_features_dimensionality.py
Test-Path .\Src\week6_src_ridge_models.py
Test-Path .\Src\week7_src_hist_gradient_boosting.py
Test-Path .\Notebooks\week7_notebook_model_training_tuning.ipynb
```

All checks should display `True`. If an earlier-week source is missing, do not run; restore it from the existing repository rather than editing Week 7 to guess.

Install the existing dependencies and validate Python syntax:

```powershell
python -m pip install -r .\requirements.txt
python -c "import numpy,pandas,sklearn,matplotlib,joblib,threadpoolctl; print('Dependencies OK')"
python -m py_compile .\Src\week7_src_hist_gradient_boosting.py
```

Optionally test code behavior in seconds on generated **synthetic test data** (not used for course metrics):

```powershell
python .\tests\test_week7_smoke.py
```

## 4. Run the complete actual Week 7 pipeline

Open `Notebooks/week7_notebook_model_training_tuning.ipynb`, select the same Python kernel used in Week 6, click **Run All**, review and save (`Ctrl+S`) without clearing outputs.

The notebook automatically:
1. Reconstructs the real Week 5 PCA inputs using the Week 6 adapter and official BTS archives.
2. Verifies chronological splitting and preserves the original training-only transformations.
3. Fits HGB_A, HGB_B, and HGB_C on the complete training set.
4. Scores each variation using *both* training and validation MAE/RMSE/R².
5. Chooses a provisional winner based on validation MAE only.
6. Saves model output, plots, CSV comparisons, and a report-ready results summary.

Expected final notebook output:

```text
All Week 7 checks passed.
===== WEEK 7 END =====
```

Alternatively, from the root, you can run **only** the standalone script:

```powershell
python .\Src\week7_src_hist_gradient_boosting.py
```

**Do not run both needlessly** — they would each rebuild and refit the full model on the large dataset. Three hist-gradient-boosted models on over a million rows may require substantial CPU resources.

## 5. Check the generated Week 7 outputs

```powershell
Get-ChildItem .\Reports\week7\week7_tables -File
Get-ChildItem .\Reports\week7\week7_figures -File
Get-ChildItem .\Models\week7 -File
Import-Csv .\Reports\week7\week7_tables\week7_three_model_comparison.csv | Format-Table -AutoSize
Get-Content .\Reports\week7\week7_verified_results_for_report.md
```

The model comparison CSV should contain exactly **three data rows**, with all six training/validation metric columns. The JSON winner should indicate `EvaluatedTestSet: false`. Cross-check the split counts and the logical winner selection. Compare Week 7 validation MAE with the historical Week 6 Ridge_A reference (3,257.18 passengers), but do not use the test partition to choose between weeks.

## 6. Finalize the written report

Open:

```text
Reports/week7/week7_report_second_modeling_approach.docx
```

In **Section 5**, copy measured numbers from `week7_three_model_comparison.csv` into the **three-row** table and remove its instructional note. In **Section 6**, insert the actual MAE/RMSE figures and discuss the A–B and B–C error differences. In **Section 7**, insert the selected model/parameters/metrics from `week7_winner_summary.json`; embed the actual prediction-scatter and monthly validation MAE figures. Remove every `[RUN]` and `REQUIRED BEFORE SUBMISSION` paragraph. Do not copy figures or metrics from the synthetic code test.

Recheck the final document for **Arial 11-point**, **1.5-spaced paragraphs**, and **10-page main-body maximum**. The draft is already formatted professionally but is **not final** until your real numerical results and images are inserted. This package deliberately includes no script to automatically generate Word documents, consistent with the previous-week workflow preference.

Upload the finalized `.docx` to Google Drive, open/convert it using Google Docs, check table/figure pagination, grant the instructor access, and copy the shareable link.

## 7. Save PDF of the **executed** notebook

Only after running and saving the notebook with outputs:

```powershell
jupyter nbconvert --to html ".\Notebooks\week7_notebook_model_training_tuning.ipynb" --output "week7_notebook_model_training_tuning" --output-dir ".\Reports\week7"
```

Open `Reports/week7/week7_notebook_model_training_tuning.html` in a browser, choose **Print → Save as PDF**, and name it:

```text
Reports/week7/week7_notebook_model_training_tuning.pdf
```

Verify the code, metrics, figures and both Week 7 START/END markers are visible. Delete the temporary HTML (not the PDF) afterward:

```powershell
Remove-Item ".\Reports\week7\week7_notebook_model_training_tuning.html"
```

## 8. Commit and push to GitHub

Run each step separately in VS Code PowerShell, checking for errors:

```powershell
git status
git add -A
git status
git diff --cached --stat
git commit -m "Week 7: train three histogram boosting variants and evaluate validation performance"
git pull --rebase origin main
git push origin main
git status
git --no-pager log --oneline --decorate -5
```

**Stop if the rebase fails; do not force push.** If there are *only* already-committed files and Git reports `nothing to commit`, skip `git commit`. The tracked files should include the final written report, executed notebook, executed notebook PDF, tables/figures, source, and selected model; Python caches and notebook checkpoints must remain ignored.

GitHub repository: `https://github.com/Laoshi88/ADAN8888-Airline-Demand-Forecasting`

Canvas submission: **GitHub URL**. Share the finalized Google Doc report link with the instructor as specified by the assignment. The Week 7 Canvas page shows **Friday, October 23, 2026 at 8:59 PM**, whereas the page's internal due-date text says **Friday at 11:59 PM**; rely on the live Canvas due field unless the instructor confirms otherwise.

## 9. Discussion 1

Use the draft in `Reports/week7/week7_discussion_initial_post.md` and add empirical winner results after the real run if appropriate. Complete the separate required peer reply only after viewing an actual classmate's post.
