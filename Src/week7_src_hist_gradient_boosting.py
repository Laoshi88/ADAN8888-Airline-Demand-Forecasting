"""ADAN8888 Week 7: Three hist-gradient-boosting regressors on the Week 5 PCA features.

Run from the root of the existing repository:
    python Src/week7_src_hist_gradient_boosting.py

This source reuses Week 6's loader to reproduce the full training-fitted Week 5
PCA data. It never fits on, scores, or selects with the final test partition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

FEATURES = ("PC1", "PC2")
TARGET = "TargetPassengersNextMonth"
SPLITS = ("Training", "Validation")

# Three controlled settings, not the draft's eight-setting Cartesian grid.
# A→B isolates the number of boosting iterations; B→C isolates leaf capacity.
CANDIDATES = (
    {"Model": "HGB_A", "learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 15},
    {"Model": "HGB_B", "learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 15},
    {"Model": "HGB_C", "learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 31},
)
COMMON = {"loss": "squared_error", "min_samples_leaf": 50,
          "l2_regularization": 1.0, "early_stopping": False, "random_state": 42}


@dataclass
class Week7Results:
    split_summary: pd.DataFrame
    comparison: pd.DataFrame
    metrics_long: pd.DataFrame
    ranking: pd.DataFrame
    winning_model_name: str
    winning_model: HistGradientBoostingRegressor
    winning_settings: dict
    source_description: str
    x_validation: np.ndarray
    y_validation: np.ndarray
    validation_months: pd.Series


def project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if (folder / "Src" / "week6_src_ridge_models.py").is_file() and (folder / "Notebooks").is_dir():
            return folder
    raise FileNotFoundError("Open the Git repository containing Src/week6_src_ridge_models.py")


def load_training_and_validation(root: Path):
    """Reuse the existing verified Week 6 data adapter; do not use 500-row PCA samples."""
    path = root / "Src" / "week6_src_ridge_models.py"
    spec = importlib.util.spec_from_file_location("week7_prior_week6_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load the Week 6 adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    data, split_summary, descriptor = module._load_week5_data(root)
    if not all(k in data for k in SPLITS):
        raise ValueError("Week 6 loader did not return Training and Validation.")
    # Assert chronology independently of prior-week implementation.
    dates_train = pd.to_datetime(data["Training"]["ForecastMonth"])
    dates_val = pd.to_datetime(data["Validation"]["ForecastMonth"])
    if dates_train.isna().any() or dates_val.isna().any() or dates_train.max() >= dates_val.min():
        raise AssertionError("Training and validation forecast months overlap or are invalid.")
    # Do not retain or inspect a test partition, even if an adapter ever returns one.
    return {k: data[k] for k in SPLITS}, split_summary, descriptor


def _features_and_target(frame: pd.DataFrame):
    required = set(FEATURES) | {TARGET, "ForecastMonth"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing expected Week 5/6 columns: {sorted(missing)}")
    x = frame[list(FEATURES)].to_numpy(dtype=np.float64, copy=True)
    y = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    months = pd.to_datetime(frame["ForecastMonth"], errors="coerce")
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Empty or nonfinite model inputs or outcomes")
    if np.any(y < 0) or months.isna().any():
        raise ValueError("Invalid passenger counts or forecast months")
    return x, y, months.reset_index(drop=True)


def measure(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.maximum(np.asarray(prediction, dtype=np.float64), 0)
    return {"MAE": float(mean_absolute_error(y, prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
            "R2": float(r2_score(y, prediction))}


def fit_three_variations(training: pd.DataFrame, validation: pd.DataFrame,
                         split_summary: pd.DataFrame, source: str = "Week 5 PCA") -> Week7Results:
    x_train, y_train, months_train = _features_and_target(training)
    x_val, y_val, months_val = _features_and_target(validation)
    if months_train.max() >= months_val.min():
        raise AssertionError("Chronological split violation")
    counts = dict(zip(split_summary["Split"], split_summary["Rows"].astype(int)))
    for split, x in (("Training", x_train), ("Validation", x_val)):
        if split in counts and len(x) != counts[split]:
            raise AssertionError(f"{split}: changed number of Week 3 observations")

    rows = []
    fitted = {}
    with threadpool_limits(limits=4):
        for settings in CANDIDATES:
            params = {k: v for k, v in settings.items() if k != "Model"}
            model = HistGradientBoostingRegressor(**params, **COMMON)
            model.fit(x_train, y_train)
            fitted[settings["Model"]] = model
            entry = {**settings, "FittedIterations": int(model.n_iter_)}
            for split, x, y in (("Training", x_train, y_train), ("Validation", x_val, y_val)):
                entry.update({f"{split}_{key}": val for key, val in measure(y, model.predict(x)).items()})
            rows.append(entry)
            print(f"Completed {settings['Model']}: validation MAE {entry['Validation_MAE']:,.2f} passengers", flush=True)

    comparison = pd.DataFrame(rows)
    if len(comparison) != 3 or not np.isfinite(comparison.select_dtypes(include="number")).all().all():
        raise AssertionError("Missing/nonfinite three-model evaluation results")
    comparison["MAE_Gap_ValidationMinusTraining"] = comparison["Validation_MAE"] - comparison["Training_MAE"]
    ranking = comparison.sort_values(
        ["Validation_MAE", "Validation_RMSE", "max_leaf_nodes", "max_iter", "Model"],
        kind="mergesort",
    ).reset_index(drop=True)
    winner = ranking.iloc[0]["Model"]
    best = next(c for c in CANDIDATES if c["Model"] == winner)
    records = []
    for _, row in comparison.iterrows():
        for split in SPLITS:
            records.append({"Model": row.Model, "learning_rate": row.learning_rate,
                            "max_iter": row.max_iter, "max_leaf_nodes": row.max_leaf_nodes,
                            "Split": split, "Rows": len(x_train) if split == "Training" else len(x_val),
                            "MAE": row[f"{split}_MAE"], "RMSE": row[f"{split}_RMSE"],
                            "R2": row[f"{split}_R2"]})
    return Week7Results(split_summary.copy(), comparison, pd.DataFrame(records), ranking,
                        winner, fitted[winner], dict(best), source, x_val, y_val, months_val)


def _plots(result: Week7Results, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    ordered = result.comparison
    for metric in ("MAE", "RMSE"):
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        labels = ordered["Model"]
        ax.plot(labels, ordered[f"Training_{metric}"], marker="o", label=f"Training {metric}")
        ax.plot(labels, ordered[f"Validation_{metric}"], marker="o", label=f"Validation {metric}")
        ax.set(title=f"Week 7: Training versus Validation {metric}",
               xlabel="Histogram gradient boosting variation", ylabel=f"{metric} (passengers)")
        ax.legend(); fig.tight_layout()
        fig.savefig(target / f"week7_figure_{metric.lower()}_by_variation.png", dpi=175)
        plt.close(fig)

    # Plot sample is independent of full data used for all model metrics.
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(len(result.y_validation), min(len(result.y_validation), 4000), replace=False))
    y = result.y_validation[indices]
    pred = np.maximum(result.winning_model.predict(result.x_validation[indices]), 0)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(y, pred, s=8, alpha=0.28)
    endpoint = max(float(y.max()), float(pred.max()), 1)
    ax.plot([0, endpoint], [0, endpoint], linestyle="--", label="Perfect prediction")
    ax.set(xlabel="Actual next-month passengers", ylabel="Predicted passengers",
           title=f"{result.winning_model_name}: Validation Predictions (sample)")
    ax.legend(); fig.tight_layout()
    fig.savefig(target / "week7_figure_validation_actual_vs_predicted.png", dpi=175)
    plt.close(fig)

    # Calendar-month error uses all validation observations; provides temporal insight.
    all_pred = np.maximum(result.winning_model.predict(result.x_validation), 0)
    monthly = pd.DataFrame({"ForecastMonth": result.validation_months,
                            "AbsoluteError": np.abs(all_pred - result.y_validation)})
    grouped = monthly.groupby("ForecastMonth", as_index=False)["AbsoluteError"].mean()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(grouped.ForecastMonth, grouped.AbsoluteError, marker="o", markersize=3)
    ax.set(title=f"{result.winning_model_name}: Validation MAE by Forecast Month",
           xlabel="Forecast month", ylabel="Monthly mean absolute error (passengers)")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(target / "week7_figure_monthly_validation_mae.png", dpi=175)
    plt.close(fig)
    return grouped


def save_week7(result: Week7Results, root: Path) -> None:
    base = root / "Reports" / "week7"
    tables = base / "week7_tables"
    figures = base / "week7_figures"
    models = root / "Models" / "week7"
    for folder in (tables, figures, models): folder.mkdir(parents=True, exist_ok=True)
    result.split_summary.to_csv(tables / "week7_split_summary.csv", index=False)
    result.metrics_long.to_csv(tables / "week7_metrics_long.csv", index=False)
    result.comparison.to_csv(tables / "week7_three_model_comparison.csv", index=False)
    result.ranking.to_csv(tables / "week7_validation_ranking.csv", index=False)
    group = _plots(result, figures)
    group.to_csv(tables / "week7_monthly_validation_mae.csv", index=False)
    best = result.ranking.iloc[0]
    metadata = {
        "SelectedModel": result.winning_model_name,
        "SelectionRule": "Lowest validation MAE; then validation RMSE; then simpler leaf/iteration configuration",
        "Hyperparameters": {k: result.winning_settings[k] for k in ("learning_rate", "max_iter", "max_leaf_nodes")},
        "FixedHyperparameters": COMMON, "InputFeatures": list(FEATURES),
        "TrainingMAE": float(best.Training_MAE), "ValidationMAE": float(best.Validation_MAE),
        "TrainingRMSE": float(best.Training_RMSE), "ValidationRMSE": float(best.Validation_RMSE),
        "TrainingR2": float(best.Training_R2), "ValidationR2": float(best.Validation_R2),
        "NegativePredictionPolicy": "Clip to zero for all evaluated models",
        "InputProvenance": result.source_description,
        "EvaluatedTestSet": False,
        "Week6PriorValidationMAEReference": 3257.1774025706486,
        "Week6ReferenceSource": "Previously supplied Week 6 notebook results; not used for Week 7 winner selection",
    }
    text = json.dumps(metadata, indent=2)
    (tables / "week7_winner_summary.json").write_text(text, encoding="utf-8")
    (models / "week7_model_manifest.json").write_text(text, encoding="utf-8")
    model_path = models / "week7_selected_hist_gradient_boosting.joblib"
    joblib.dump(result.winning_model, model_path)
    restored = joblib.load(model_path)
    if not np.allclose(restored.predict(result.x_validation[:16]),
                       result.winning_model.predict(result.x_validation[:16])):
        raise AssertionError("Model serialization check failed")

    lines = [
        "# Week 7 — empirical results for the written report",
        "", "These values were generated by executing the **real** Week 7 pipeline.",
        "", f"Selected: **{metadata['SelectedModel']}**, hyperparameters: {metadata['Hyperparameters']}",
        f"Validation MAE: **{metadata['ValidationMAE']:,.2f}** passengers. The held-out test set was NOT evaluated.",
        "", "| Model | Learning rate | Iterations | Leaf nodes | Train MAE | Val MAE | Train RMSE | Val RMSE | Train R² | Val R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in result.comparison.iterrows():
        lines.append(f"| {row.Model} | {row.learning_rate:.2f} | {int(row.max_iter)} | {int(row.max_leaf_nodes)} "
                     f"| {row.Training_MAE:,.2f} | {row.Validation_MAE:,.2f} "
                     f"| {row.Training_RMSE:,.2f} | {row.Validation_RMSE:,.2f} "
                     f"| {row.Training_R2:.4f} | {row.Validation_R2:.4f} |")
    lines.append("\nUse the generated figures under `week7_figures/`; do not include synthetic QA results in the report.\n")
    (base / "week7_verified_results_for_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_week7(root: Path | None = None) -> Week7Results:
    root = root or project_root()
    data, split_summary, source = load_training_and_validation(root)
    result = fit_three_variations(data["Training"], data["Validation"], split_summary, source)
    save_week7(result, root)
    return result


def print_summary(result: Week7Results) -> None:
    print("===== WEEK 7 START =====")
    print("Modeling approach: HistGradientBoostingRegressor (PC1, PC2)")
    print("Test data scored: NO")
    print("\nChronological partitions:")
    print(result.split_summary.to_string(index=False))
    print("\nAll training and validation metrics:")
    print(result.comparison.round(3).to_string(index=False))
    print(f"\nProvisional winner: {result.winning_model_name}")
    print("All Week 7 checks passed.")
    print("===== WEEK 7 END =====")


if __name__ == "__main__":
    print("Beginning Week 7 modeling execution...", flush=True)
    print_summary(run_week7())
