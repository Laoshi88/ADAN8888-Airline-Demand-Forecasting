# Week 11 — Explaining airline forecasts and evaluating bias

## What is included in the merge

- `Src/week11_scr_forecast_explainability_and_bias_audit.py` and `Notebooks/week11_notebook_forecast_explainability_and_bias_audit.ipynb` — audit code and executed notebook for feature importance, five individual forecasts, subgroup errors, and model retraining.
- `Data/processed/week11/five_random_test_rows.csv` — the five sampled carrier–route–months used to explain individual predictions.
- `Reports/week11/tables/` and `Reports/week11/figures/` — result tables and the validation feature-importance chart.
- `Reports/week11/week11_report_forecast_explainability_and_bias_audit` — final report in Word and PDF, executed notebook PDF, discussion post, and reading notes.
- `requirements_week11.txt` — dependencies for reproducing the audit with Python 3.12.

## Modeling design and verified results

- **Benchmark.** The audit uses the Week 10 model, `MLP_B_temporal`, with 16 inputs. Each forecast estimates next-month passengers for one carrier and directed airport pair.
- **Feature importance.** Shuffling the one-month passenger lag increased validation sample MAE by **2,238.98 passengers**, the largest increase for any single input.
- **Five forecasts.** Changing the prior-year passenger count moved all five forecasts by at least 50 passengers and 10% of the original forecast. Only DEN–RSW and IAD–DCA moved closer to actual traffic.
- **Route history.** Historical test observations missing prior-year history had lower mean absolute error (MAE) than observations with history (**556 vs. 831 passengers**). Their errors represented a larger share of actual traffic (**37.06% vs. 13.48%** volume-weighted absolute percentage error, or WAPE).
- **Airport coverage.** In a separate comparison, test WAPE was **21.86%** at origins with less training coverage and **13.66%** at origins with more coverage.
- **Refit decision.** Giving missing-history records more weight lowered historical test MAE but raised validation MAE. Retain the Week 10 benchmark.

| MAE in passengers | Training | Validation | Historical test |
| --- | ---: | ---: | ---: |
| Week 10 benchmark | 808.17 | 809.33 | 790.29 |
| Reweighted model | 803.66 | 813.94 | 779.51 |

## Step-by-step VS Code / PowerShell workflow

1. **Check prerequisites** from the repository root. The archive count should be 12 and both file checks should return `True`. The audit also uses the existing Week 2, 3, 5, and 10 source files.

   ```powershell
   (Get-ChildItem .\Data\raw\*all_carriers_*.zip -File).Count
   Test-Path .\Models\week10\week10_selected_enhanced_mlp_bundle.joblib
   Test-Path .\Reports\week10\week10_tables\week10_train_validation_test_metrics.csv
   ```

2. **Review the completed run.** Open `Reports/week11/week11_report_forecast_explainability_and_bias_audit.pdf` for the findings or `Reports/week11/week11_notebook_forecast_explainability_and_bias_audit.pdf` for the code and outputs. These commands display the six model comparisons and five forecast cases.

   ```powershell
   Import-Csv .\Reports\week11\tables\metrics.csv | Format-Table -AutoSize
   Import-Csv .\Reports\week11\tables\cases.csv | Format-Table -AutoSize
   ```

3. **Reproduce the audit if needed.** Open `Notebooks/week11_notebook_forecast_explainability_and_bias_audit.ipynb` in VS Code, select the Week 11 Python 3.12 kernel, and choose **Run All**. The final cell should print:

   ```text
   Audit complete. Full-partition baseline reconciled on every split.
   ```

## Interpretation safeguards

- The historical test period was examined in earlier weeks. A later, unseen period is needed to evaluate future performance.
- T-100 contains aggregate service records. The audit measures differences by route history and airport coverage. Estimating protected-class disparities would require demographic evidence.
- Reweighting reduced overall historical test overprediction from **177.73 to 46.10 passengers**, while overprediction for missing-history observations rose from **164.61 to 184.87**. Review subgroup errors before using forecasts to justify service reductions.

Repository: https://github.com/Laoshi88/ADAN8888-Airline-Demand-Forecasting