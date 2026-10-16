"""ADAN8888 Week 6 — first forecasting model (Ridge), three alpha settings.

The earlier Week 5 PCA pipeline is reused *as-is*, preserving the Week 3
forecast-month split and training-only preprocessing. Validation alone selects
this week's winner. Test labels are NEVER used for scoring or selection.

Run from the repository root: python Src/week6_src_ridge_models.py
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_SETTINGS = [
    ("Ridge_A", 1.0),
    ("Ridge_B", 10_000.0),
    ("Ridge_C", 1_000_000.0),
]
TARGET = "TargetPassengersNextMonth"
FEATURES = ["PC1", "PC2"]


@dataclass
class Week6Results:
    split_summary: pd.DataFrame
    metrics_long: pd.DataFrame
    metrics_table: pd.DataFrame
    comparison: pd.DataFrame
    winner: str
    winner_alpha: float
    winning_model: Ridge
    data_source: str
    x_train: np.ndarray
    x_validation: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray


def detect_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for p in (start, *start.parents):
        if all((p / f).exists() for f in ["Src", "Notebooks", "Data"]):
            return p
    raise FileNotFoundError("Open the ADAN8888 repository root (contains Src, Notebooks, Data).")


def _import_file(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Required earlier-week code not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_week5_data(root: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, str]:
    """Recreate the Week 5 *full* PCA features, not its 500-row GitHub samples.

    This deliberately uses the training-only Week 5 fit of medians, scaling,
    PCA. Feature creation never uses t+1 passenger targets.
    """
    wk5 = _import_file(root / "Src" / "week5_src_features_dimensionality.py", "week6_week5_pipeline")
    monthly = wk5.load_monthly_without_rewriting_reports(root)
    features = wk5.create_cross_field_features(monthly)
    partitions, split_summary = wk5.get_week3_splits(features, root)
    fitted = wk5.fit_feature_reduction(partitions, split_summary)
    # Week 5 exports only small preview CSVs; redo *frozen transforms* on entire splits.
    processed: dict[str, pd.DataFrame] = {}
    for split in ["Training", "Validation"]:
        source = partitions[split]
        logged = wk5._numeric_inputs(source)  # deterministic log1p of nonnegative inputs
        filled = logged.fillna(fitted.medians)
        matrix = fitted.scaler.transform(filled[fitted.kept_features].to_numpy(dtype="float64"))
        pcs = fitted.pca.transform(matrix)
        if pcs.shape != (len(source), 2) or not np.isfinite(pcs).all():
            raise ValueError(f"Bad PCA transformation for {split}.")
        out = pd.DataFrame(pcs, columns=FEATURES)
        out[TARGET] = pd.to_numeric(source[TARGET], errors="coerce").to_numpy()
        out["ForecastMonth"] = pd.to_datetime(source["TargetPeriod"], errors="coerce").to_numpy()
        processed[split] = out
    # Week 5's actual training + validation rows remain unchanged.
    counts = dict(zip(split_summary["Split"], split_summary["Rows"].astype(int)))
    for split in processed:
        if len(processed[split]) != counts[split]:
            raise AssertionError(f"{split} row count changed vs Week 3/5.")
    if processed["Training"]["ForecastMonth"].max() >= processed["Validation"]["ForecastMonth"].min():
        raise AssertionError("Forecast periods are not chronologically separated.")
    return processed, split_summary, "Week 5 source (reconstructed full PCA from official BTS ZIPs)"


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Metrics on original passenger-count scale; negatives clipped at 0."""
    pred = np.maximum(np.asarray(pred, dtype="float64"), 0.0)
    if not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite passenger actuals or predictions")
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def train_and_compare(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    split_summary: pd.DataFrame,
    data_source: str = "Week 5 PCA representation",
) -> Week6Results:
    """One modeling family, exactly 3 hyperparameter variants; validation selects winner."""
    x_train = train[FEATURES].to_numpy(dtype="float64")
    x_val = validation[FEATURES].to_numpy(dtype="float64")
    y_train = pd.to_numeric(train[TARGET], errors="coerce").to_numpy(dtype="float64")
    y_val = pd.to_numeric(validation[TARGET], errors="coerce").to_numpy(dtype="float64")
    if min(len(x_train), len(x_val)) == 0:
        raise ValueError("Training or validation split is empty")
    if not all(np.isfinite(x).all() for x in [x_train, x_val, y_train, y_val]):
        raise ValueError("Nonfinite model inputs or labels")
    if (y_train < 0).any() or (y_val < 0).any():
        raise ValueError("Negative observed passenger counts are invalid")

    rows = []
    models = {}
    for label, alpha in MODEL_SETTINGS:
        model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
        model.fit(x_train, y_train)
        models[label] = model
        for split, x, y in (("Training", x_train, y_train), ("Validation", x_val, y_val)):
            errors = _metrics(y, model.predict(x))
            rows.append({"Model": label, "Alpha": alpha, "Split": split, "Rows": len(y), **errors})
    long = pd.DataFrame(rows)
    wide = pd.DataFrame([
        {"Model": label, "Alpha": alpha,
         **{f"{split}_{metric}": float(long.loc[(long.Model == label) & (long.Split == split), metric].iloc[0])
            for split in ("Training", "Validation") for metric in ("MAE", "RMSE", "R2")}}
        for label, alpha in MODEL_SETTINGS
    ])
    wide["MAE_Gap_ValidationMinusTraining"] = wide["Validation_MAE"] - wide["Training_MAE"]
    # MAE is selected a priori. RMSE then smaller alpha for exact ties.
    sorted_models = wide.sort_values(["Validation_MAE", "Validation_RMSE", "Alpha"], kind="mergesort")
    winner = str(sorted_models.iloc[0]["Model"])
    alpha = float(sorted_models.iloc[0]["Alpha"])
    comparison = sorted_models[["Model", "Alpha", "Validation_MAE", "Validation_RMSE", "Validation_R2", "MAE_Gap_ValidationMinusTraining"]].copy()
    return Week6Results(split_summary.copy(), long, wide, comparison, winner, alpha, models[winner],
                        data_source, x_train, x_val, y_train, y_val)


def save_outputs(results: Week6Results, root: Path) -> None:
    reports = root / "Reports" / "week6"
    tables = reports / "week6_tables"
    figures = reports / "week6_figures"
    model_dir = root / "Models" / "week6"
    for p in (tables, figures, model_dir):
        p.mkdir(parents=True, exist_ok=True)
    results.split_summary.to_csv(tables / "week6_split_summary.csv", index=False)
    results.metrics_long.to_csv(tables / "week6_metrics_train_validation_long.csv", index=False)
    results.metrics_table.to_csv(tables / "week6_models_metrics_comparison.csv", index=False)
    results.comparison.to_csv(tables / "week6_validation_ranking.csv", index=False)
    selected = results.metrics_table.loc[results.metrics_table.Model.eq(results.winner)].iloc[0]
    winner_json = {
        "SelectedModel": results.winner, "RidgeAlpha": results.winner_alpha,
        "PrimarySelectionMetric": "Lowest validation MAE, with RMSE/alpha as tie-breakers",
        "ValidationMAE": float(selected.Validation_MAE),
        "ValidationRMSE": float(selected.Validation_RMSE),
        "ValidationR2": float(selected.Validation_R2),
        "TrainMAE": float(selected.Training_MAE),
        "TrainRMSE": float(selected.Training_RMSE),
        "TrainR2": float(selected.Training_R2),
        "InputFeatures": FEATURES,
        "PredictionPostprocessing": "clip at 0 passengers for business-plausible counts",
        "OutcomeScale": "original passengers, not log scale",
        "DataSource": results.data_source,
        "TestUsedForFittingOrSelection": False,
    }
    (tables / "week6_winner_summary.json").write_text(json.dumps(winner_json, indent=2), encoding="utf-8")
    joblib.dump(results.winning_model, model_dir / "week6_selected_ridge.joblib")
    (model_dir / "week6_model_manifest.json").write_text(json.dumps(winner_json, indent=2), encoding="utf-8")
    # Check exact model serialization round-trip on 16 validation examples.
    loaded = joblib.load(model_dir / "week6_selected_ridge.joblib")
    if not np.allclose(loaded.predict(results.x_validation[:16]), results.winning_model.predict(results.x_validation[:16])):
        raise AssertionError("Serialized model predictions changed")

    fig, ax = plt.subplots(figsize=(8, 5))
    names = results.metrics_table["Model"].tolist()
    ax.plot(names, results.metrics_table["Training_MAE"], marker="o", label="Training MAE")
    ax.plot(names, results.metrics_table["Validation_MAE"], marker="o", label="Validation MAE")
    ax.set_title("Week 6 Ridge Regularization: Training vs Validation MAE")
    ax.set_ylabel("Mean absolute error (passengers)")
    ax.set_xlabel("Alpha settings: 1 / 10,000 / 1,000,000")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "week6_figure_mae_by_alpha.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(names, results.metrics_table["Training_RMSE"], marker="o", label="Training RMSE")
    ax.plot(names, results.metrics_table["Validation_RMSE"], marker="o", label="Validation RMSE")
    ax.set_title("Week 6 Ridge Regularization: Training vs Validation RMSE")
    ax.set_ylabel("Root mean squared error (passengers)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "week6_figure_rmse_by_alpha.png", dpi=180)
    plt.close(fig)

    # Fixed-size reproducible validation sample; no full validation predictions uploaded.
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(results.y_validation), size=min(len(results.y_validation), 4_000), replace=False))
    expected = results.y_validation[idx]
    predicted = np.maximum(results.winning_model.predict(results.x_validation[idx]), 0.0)
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.scatter(expected, predicted, s=8, alpha=0.28)
    high = max(float(expected.max()), float(predicted.max()), 1.0)
    ax.plot([0, high], [0, high], linestyle="--", linewidth=1, label="Perfect forecast")
    ax.set(xlabel="Actual next-month passengers", ylabel="Predicted next-month passengers",
           title=f"Selected {results.winner}: Validation Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "week6_figure_validation_actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    insert_lines = [
        "# Week 6 verified results — paste into written report only after reviewing",
        "", "Results based on actual execution of Week 6 on the repository dataset.", "",
        f"Selected {results.winner} (Ridge alpha = {results.winner_alpha:,.0f}) because it attained the lowest validation MAE.",
        "", "| Variant | Alpha | Train MAE | Val MAE | Train RMSE | Val RMSE | Train R² | Val R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in results.metrics_table.iterrows():
        insert_lines.append(f"| {row.Model} | {row.Alpha:,.0f} | {row.Training_MAE:,.2f} | {row.Validation_MAE:,.2f} | {row.Training_RMSE:,.2f} | {row.Validation_RMSE:,.2f} | {row.Training_R2:.4f} | {row.Validation_R2:.4f} |")
    insert_lines += ["", "Figures saved under Reports/week6/week6_figures/.",
                     "The test partition was NOT evaluated, and must not be used to choose the Week 6 winner."]
    (reports / "week6_verified_results_for_report.md").write_text("\n".join(insert_lines)+"\n", encoding="utf-8")


def run_week6(root: Path | None = None) -> Week6Results:
    root = root or detect_project_root()
    partitions, split_summary, source = _load_week5_data(root)
    results = train_and_compare(partitions["Training"], partitions["Validation"], split_summary, source)
    save_outputs(results, root)
    return results


def print_summary(results: Week6Results) -> None:
    print("===== WEEK 6 START =====")
    print("Method: Ridge regression on Week 5 PC1 and PC2")
    print("Test partition evaluated: NO")
    print("Chronological split:")
    print(results.split_summary.to_string(index=False))
    print("\nTraining/validation metrics for all three alpha settings:")
    print(results.metrics_table.round(4).to_string(index=False))
    print(f"\nSelected Week 6 winner: {results.winner}, alpha={results.winner_alpha:,.0f} (validation MAE)")
    print("All Week 6 checks passed.")
    print("===== WEEK 6 END =====")


if __name__ == "__main__":
    print_summary(run_week6())
