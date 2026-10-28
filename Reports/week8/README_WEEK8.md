# Week 8 — Third Modeling Approach: Multilayer Perceptron Neural Network

## What is included in merge

- `Src/week8_src_mlp_neural_network.py` — reusable Week 8 neural-network training, evaluation, ranking, and model-export source.
- `Notebooks/week8_notebook_mlp_model_training_tuning.ipynb` — documented Jupyter workflow calling the same reusable source.
- `Reports/week8/week8_report_third_modeling_approach.docx` — formatted written report.
- `tests/test_week8_smoke.py` — optional synthetic-data-only code test.
- `Models/week8/README.md` — saved-model inference requirements.
- `requirements_week8_additions.txt` — required dependencies beyond existing project environment.

## Modeling design

- **Method**: feedforward Multi-Layer Perceptron neural-network regression (`MLPRegressor`) on Week 5 `PC1`/`PC2`. A third model family distinct from Weeks 6–7.
- **Three hyperparameter variants**: `MLP_A` hidden layers `(32,16)`, `MLP_B` `(64,32)`, and `MLP_C` `(64,32,16)`.
- **Fixed settings**: ReLU, Adam solver, `alpha=0.001`, `learning_rate_init=0.001`, `batch_size=4096`, `max_iter=45`, `random_state=42`, `early_stopping=False`.
- **Training-fitted preprocessing**: `StandardScaler` for PC1/PC2 and separate training-fitted target `StandardScaler` inside `TransformedTargetRegressor`. Week 5 prior transformation remains training-fitted too.
- **Metrics**: Training and validation MAE, RMSE, R² in **passenger units**, with zero floor for negative predicted passenger counts.
- **Winner criterion**: *lowest validation MAE*, then validation RMSE, then fewer trainable parameters for exact ties. No training/validation data combined for fitting in Week 8.
- **Test set**: not used or evaluated; saved for a later final model assessment after model selection.
- **Full data**: Uses the original BTS `Data/raw/` ZIPs and the full Week 5 PCA pipeline via Week 6's adapter. It does **not** evaluate only Week 5 sample/preview rows.
- **Week 7 reference**: HGB_C had a validation MAE of 1,161.07 passengers. This is a reference only; it must not change the Week 8 three-way selection.
- **Computing**: scikit-learn's MLP uses CPU (no GPU support). The full-data training may be compute-intensive.

## VS Code / PowerShell workflow

1. Verify prerequisites. All checks should be `True`:

   ```powershell
   Test-Path .\Src\week2_src_t100_ingest_explore.py
   Test-Path .\Src\week3_src_eda_split.py
   Test-Path .\Src\week5_src_features_dimensionality.py
   Test-Path .\Src\week6_src_ridge_models.py
   Test-Path .\Src\week8_src_mlp_neural_network.py
   Test-Path .\Notebooks\week8_notebook_mlp_model_training_tuning.ipynb
   ```

   Confirm the official BTS annual ZIP files are present under `Data/raw/`. Keep that source data unchanged.

2. Install/check dependencies, and test the Week 8 Python source:

   ```powershell
   python -m pip install -r .\requirements.txt
   python -m pip install -r .\requirements_week8_additions.txt
   python -c "import pandas,numpy,sklearn,matplotlib,joblib,threadpoolctl,nbformat; print('Week 8 dependencies OK')"
   python -m py_compile .\Src\week8_src_mlp_neural_network.py
   ```

   Optional small synthetic functionality test (**not real project metrics**):

   ```powershell
   python .\tests\test_week8_smoke.py
   ```

3. Verify the generated outputs:

   ```powershell
   Get-ChildItem .\Reports\week8\week8_tables -File
   Get-ChildItem .\Reports\week8\week8_figures -File
   Get-ChildItem .\Models\week8 -File
   Import-Csv .\Reports\week8\week8_tables\week8_three_model_comparison.csv | Format-Table -AutoSize
   Get-Content .\Reports\week8\week8_verified_results_for_report.md
   ```

   Comparison CSV contains **three rows** and the long-format CSV **six rows**. 
   Check all six training/validation metrics, the `week8_winner_summary.json` winner, and `EvaluatedTestSet: false`.
