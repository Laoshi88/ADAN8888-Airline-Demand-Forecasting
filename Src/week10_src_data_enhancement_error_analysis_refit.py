"""ADAN8888 Week 10: training-only error audit, three data improvements, fixed MLP_B.

Run from the existing repository root:
  python Src/week10_src_data_enhancement_error_analysis_refit.py

Source reuses the exact saved Week 9 MLP_B reference, audits three sequential
feature interventions on original validation months, selects a model on VALIDATION
ONLY, locks that choice, and only then assesses the selected model on the Week 9
historical test period. 

- Test months were already studied in Week 9. 
- Results are not a fresh independent holdout. 
- No real experiment results are bundled here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import warnings
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # Allow plots when running without a graphical desktop.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


TARGET = "TargetPassengersNextMonth"
REFERENCE_TIME = "Period"
FORECAST_TIME = "TargetPeriod"
KEYS = ["UniqueCarrier", "Origin", "Dest"]
SPLITS = ("Training", "Validation", "Test")
RANDOM_SEED = 42
HIDDEN_LAYERS = (64, 32)
BASELINE_NAME = "MLP_B_original_PCA"
STAGE_NAMES = ("MLP_B_temporal", "MLP_B_temporal_quality", "MLP_B_enhanced_refit")
REFERENCE_VAL_MAE = 1143.1951956789312
REFERENCE_VAL_RMSE = 1988.7779681224358
REFERENCE_EXPECTED_COUNTS = {"Training": 1062603, "Validation": 239395, "Test": 264291}
BASELINE_ARTIFACT = "Models/week8/week8_selected_mlp_neural_network.joblib"
HISTORY_LOGS = ["PassengerLag1", "PassengerLag2", "PassengerLag11", "PassengerLag12", "PassengerRolling3Mean"]
ORIGINAL_LOGS = ["Passengers", "Seats", "DepScheduled", "DepPerformed", "Distance"]
LOG_COLUMNS = [
    "Passengers", "Seats", "DepScheduled", "DepPerformed", "Distance",
    "PassengerLag1", "PassengerLag2", "PassengerLag11", "PassengerLag12",
    "PassengerRolling3Mean",
]
RATIO_COLUMNS = ["LoadFactorClean", "ScheduleCompletionRate", "MonthlyLogChange", "AnnualLogChange"]


def load_source(path: Path, name: str):
    """Import existing project functions without running their main/report blocks."""
    if not path.is_file():
        raise FileNotFoundError(f"Required earlier-week source is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Dataclasses in the earlier modules need this.
    spec.loader.exec_module(module)
    return module


def detect_project_root(start: Path | None = None) -> Path:
    """Accept an explicit root or locate the existing Src/Data repository."""
    starts = [start] if start is not None else [Path.cwd(), Path(__file__).parent]
    for initial in starts:
        for folder in (initial.resolve(), *initial.resolve().parents):
            if (folder / "Src" / "week5_src_features_dimensionality.py").is_file() and (folder / "Data" / "raw").is_dir():
                return folder
    raise FileNotFoundError("Run in the original project, or pass --root PATH containing Src and Data/raw.")


def assert_split_integrity(partitions: dict, summary: pd.DataFrame) -> None:
    """Preserve Week 3's eligible rows and forecast-month 70/15/15 allocation."""
    expected = dict(zip(summary["Split"].astype(str), summary["Rows"].astype(int)))
    for split in SPLITS:
        frame = partitions[split]
        y = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(dtype=float)
        if len(frame) != expected[split] or len(frame) < 2:
            raise ValueError(f"{split}: changed row count or insufficient rows for R2.")
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError(f"{split}: targets must be observed, finite, nonnegative counts.")
        if frame.duplicated(KEYS + [REFERENCE_TIME]).any():
            raise ValueError(f"{split}: duplicate carrier-route-month keys.")
        ref = pd.to_datetime(frame[REFERENCE_TIME], errors="raise")
        dates = pd.to_datetime(frame[FORECAST_TIME], errors="raise")
        if ref.isna().any() or dates.isna().any() or not dates.eq(ref + pd.DateOffset(months=1)).all():
            raise ValueError(f"{split}: invalid one-month forecast dates.")
    for before, after in zip(SPLITS[:-1], SPLITS[1:]):
        if partitions[before][FORECAST_TIME].max() >= partitions[after][FORECAST_TIME].min():
            raise ValueError("Chronological partitions overlap.")


def split_signatures(partitions: dict) -> dict[str, str]:
    """Verify the same observations, target values, and row order after enrichment."""
    columns = KEYS + [REFERENCE_TIME, FORECAST_TIME, TARGET]
    return {name: hashlib.sha256(pd.util.hash_pandas_object(partitions[name][columns], index=False).to_numpy().tobytes()).hexdigest()
            for name in SPLITS}


def fit_original_pca(training: pd.DataFrame, week5) -> dict[str, Any]:
    """Reproduce Week 5's exact log/median/scaler/two-PC recipe on training only.

    Avoid calling its report/preview generator or making extra copies of the
    validation/test matrices. These are the same fitted transformations used
    by the original full-data adapters in Weeks 6-9.
    """
    logged = week5._numeric_inputs(training)
    medians = logged.median().fillna(0.0)
    filled = logged.fillna(medians)
    kept = filled.columns[filled.nunique(dropna=False) > 1].tolist()
    if len(kept) <= 2:
        raise ValueError("At least three varying training predictors are needed for the original PCA.")
    scaler = StandardScaler()
    matrix = scaler.fit_transform(filled[kept].to_numpy(dtype="float64"))
    pca = PCA(n_components=2, svd_solver="full").fit(matrix)
    return {"medians": medians, "kept_features": kept, "scaler": scaler, "pca": pca}


def original_pca_inputs(frame: pd.DataFrame, state: dict, week5) -> np.ndarray:
    """Transform any split with the unchanged, training-fitted reference recipe."""
    filled = week5._numeric_inputs(frame).fillna(state["medians"])
    matrix = state["scaler"].transform(filled[state["kept_features"]].to_numpy(dtype="float64"))
    pcs = state["pca"].transform(matrix)
    if not np.isfinite(pcs).all():
        raise ValueError("The reference PCA produced nonfinite values.")
    return pcs


# ---------------------------------------------------------------------------
# 1. DATA IMPROVEMENTS AND ENHANCEMENTS -- applied to ALL THREE datasets
# ---------------------------------------------------------------------------
def add_history_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Add only information available at reference month t, before eligibility.

    Exact calendar joins avoid treating the previous observed row as the
    previous month when a route has gaps. Build on the entire monthly history
    because an older row may lack its own target but still be a valid lag.
    Joining history across split boundaries is valid in this rolling setting:
    earlier observed passengers are inputs; next-month target columns are not.
    """
    out = monthly.copy()
    out[REFERENCE_TIME] = pd.to_datetime(out[REFERENCE_TIME], errors="raise")
    # Repair feature values deterministically without changing or dropping targets.
    for field in ("Passengers", "Seats", "DepScheduled", "DepPerformed", "Distance"):
        if field not in out:
            raise ValueError(f"Missing required reference-month feature: {field}")
        value = pd.to_numeric(out[field], errors="coerce")
        out[field] = value.where(value.ge(0) & np.isfinite(value))
    if out.duplicated(KEYS + [REFERENCE_TIME]).any():
        raise ValueError("Duplicate keys would make calendar-lag joins ambiguous.")
    base = out[KEYS + [REFERENCE_TIME, "Passengers"]].copy()
    for lag in (1, 2, 11, 12):
        history = base.copy()
        history[REFERENCE_TIME] += pd.DateOffset(months=lag)
        history = history.rename(columns={"Passengers": f"PassengerLag{lag}"})
        out = out.merge(history, on=KEYS + [REFERENCE_TIME], how="left", sort=False, validate="one_to_one")

    # All three exact months must exist; do not silently average partial history.
    window = out[["Passengers", "PassengerLag1", "PassengerLag2"]]
    out["PassengerRolling3Mean"] = window.mean(axis=1).where(window.notna().all(axis=1))
    # Lag 11 is the prior-year observation for TARGET month t+1: (t+1)-12=t-11.
    out["MonthlyLogChange"] = np.log1p(out["Passengers"]) - np.log1p(out["PassengerLag1"])
    out["AnnualLogChange"] = np.log1p(out["Passengers"]) - np.log1p(out["PassengerLag12"])
    load = out["Passengers"].div(out["Seats"].where(out["Seats"].gt(0)))
    out["LoadFactorClean"] = load.where(load.between(0, 1))
    # Performed departures may exceed scheduled departures; preserve such reports.
    out["ScheduleCompletionRate"] = out["DepPerformed"].div(out["DepScheduled"].where(out["DepScheduled"].gt(0)))
    out[LOG_COLUMNS + RATIO_COLUMNS] = out[LOG_COLUMNS + RATIO_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out


def category_values(frame: pd.DataFrame) -> dict[str, pd.Series]:
    values = {c: frame[c].astype("string").fillna("__MISSING__") for c in KEYS}
    values["Route"] = values["Origin"] + "->" + values["Dest"]
    return values


def unfilled_enhanced_inputs(frame: pd.DataFrame, pcs: np.ndarray, state: dict) -> pd.DataFrame:
    """Create an explicit predictor allowlist; target columns cannot enter X."""
    x = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=frame.index)
    for col in LOG_COLUMNS:
        x[f"{col}Log1p"] = np.log1p(frame[col])
        x[f"{col}Missing"] = frame[col].isna().astype(float)
    for col in RATIO_COLUMNS:
        x[col] = frame[col]
        x[f"{col}Missing"] = frame[col].isna().astype(float)
    month = pd.to_datetime(frame[FORECAST_TIME])
    reference = pd.to_datetime(frame[REFERENCE_TIME])
    x["ForecastMonthSin"] = np.sin(2 * np.pi * month.dt.month / 12)
    x["ForecastMonthCos"] = np.cos(2 * np.pi * month.dt.month / 12)
    x["ForecastDaysRatio"] = month.dt.days_in_month / reference.dt.days_in_month
    origin = state["trend_origin"]
    x["TrendMonth"] = (month.dt.year - origin.year) * 12 + month.dt.month - origin.month
    for col, values in category_values(frame).items():
        # Frequency encoding uses observed training identities, not target means.
        x[f"{col}TrainingFrequency"] = values.map(state["frequencies"][col]).fillna(0.0).astype(float)
    return x.replace([np.inf, -np.inf], np.nan)


def fit_enhancement_state(training: pd.DataFrame, pcs: np.ndarray) -> dict:
    """Learn category maps, trend origin, and imputation medians on TRAINING ONLY."""
    state = {
        "trend_origin": pd.Timestamp(training[FORECAST_TIME].min()),
        "frequencies": {c: values.value_counts(normalize=True).to_dict()
                        for c, values in category_values(training).items()},
    }
    raw = unfilled_enhanced_inputs(training, pcs, state)
    state["features"] = list(raw.columns)
    # An entirely missing training feature uses zero plus its explicit missing flag.
    state["medians"] = raw.median().fillna(0.0)
    return state


def transform_enhanced(frame: pd.DataFrame, pcs: np.ndarray, state: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the same frozen enhancement and missing-value rules to any split."""
    raw = unfilled_enhanced_inputs(frame, pcs, state)
    if list(raw.columns) != state["features"]:
        raise ValueError("Enhanced feature order changed.")
    audit = pd.DataFrame({"Feature": raw.columns, "MissingBeforeImputation": raw.isna().sum().to_numpy()})
    x = raw.fillna(state["medians"]).astype("float64")
    audit["MissingAfterImputation"] = x.isna().sum().to_numpy()
    if len(x) != len(frame) or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Enhancement changed row counts or left nonfinite inputs.")
    return x, audit


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """Evaluate on the original passenger scale, with zero-safe aggregate errors."""
    y = np.asarray(actual, dtype=float)
    pred = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    if y.ndim != 1 or pred.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Invalid actual/predicted arrays.")
    residual = y - pred  # Positive bias means systematic underprediction.
    total = float(y.sum())
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
        "MeanResidual": float(residual.mean()),
        "WAPE_Percent": float(100 * np.abs(residual).sum() / total) if total > 0 else None,
    }


# ---------------------------------------------------------------------------
# 2. MODEL ERROR ANALYSIS -- TRAINING DATASET ONLY
# ---------------------------------------------------------------------------
def training_error_analysis(training: pd.DataFrame, prediction: np.ndarray, name: str,
                            tables: Path, figures: Path) -> pd.DataFrame:
    """Locate high-error months, carriers, routes, and demand bands in training.

    These are in-sample residuals, which can be optimistic. They identify
    patterns to investigate; they do not estimate future generalization error.
    No difficult rows or genuine demand peaks are removed because of errors.
    """
    detail = training[KEYS + [REFERENCE_TIME, FORECAST_TIME, TARGET]].copy().reset_index(drop=True)
    y = detail[TARGET].to_numpy(dtype=float)
    detail["Predicted"] = np.maximum(prediction, 0)
    detail["Residual"] = y - detail["Predicted"]
    detail["AbsoluteError"] = detail["Residual"].abs()
    detail["SquaredError"] = detail["Residual"] ** 2
    detail["Route"] = detail["Origin"].astype(str) + "->" + detail["Dest"].astype(str)
    detail["DemandBand"] = pd.cut(y, [-np.inf, 0, 1000, 10000, 50000, np.inf],
                                  labels=["Zero", "1-1000", "1001-10000", "10001-50000", "Above50000"])
    monthly = None
    for group in (FORECAST_TIME, "UniqueCarrier", "Route", "DemandBand"):
        result = detail.groupby(group, observed=True, dropna=False).agg(
            Rows=("Residual", "size"), MeanActual=(TARGET, "mean"),
            MAE=("AbsoluteError", "mean"), MSE=("SquaredError", "mean"),
            MeanResidual=("Residual", "mean"),
        ).reset_index()
        result["RMSE"] = np.sqrt(result.pop("MSE"))
        result.to_csv(tables / f"week10_{name}_training_errors_by_{group}.csv", index=False)
        if group == FORECAST_TIME:
            monthly = result
    detail.nlargest(100, "AbsoluteError").to_csv(tables / f"week10_{name}_largest_training_errors.csv", index=False)
    detail["Residual"].describe(percentiles=[.01, .05, .5, .95, .99]).to_csv(tables / f"week10_{name}_training_residual_summary.csv")

    # Sampling affects scatter display only; every error table uses all training rows.
    indices = np.random.default_rng(RANDOM_SEED).choice(len(detail), min(4000, len(detail)), replace=False)
    sample = detail.iloc[indices]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(detail["Residual"], bins=80)
    axes[0].set(xlabel="Actual minus predicted passengers", ylabel="Training observations", title="Training residual distribution")
    axes[1].scatter(sample["Predicted"], sample["Residual"], alpha=.25, s=8)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Predicted passengers", ylabel="Actual minus predicted", title="Training residuals vs fitted (display sample)")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(figures / f"week10_{name}_training_residuals.png", dpi=160)
    plt.close(fig)
    return monthly


# ---------------------------------------------------------------------------
# 3. MODEL REFITTING -- NEW TRAINING DATASET ONLY
# ---------------------------------------------------------------------------
def fit_fixed_mlp(week8, x: np.ndarray, y: np.ndarray) -> tuple[Any, dict]:
    """Build a fresh MLP_B; its input and target scalers fit on training only.

    Reusing Week 8's factory preserves architecture, regularization, optimizer,
    seed, epoch limit, and early_stopping=False. The latter avoids an implicit
    random validation split inside this chronological forecasting experiment.
    """
    model = week8.build_model(HIDDEN_LAYERS)
    params = model.regressor.named_steps["mlp"].get_params()
    if params["early_stopping"] or params["hidden_layer_sizes"] != HIDDEN_LAYERS:
        raise ValueError("Week 8 no longer matches the fixed MLP_B recipe.")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    network = model.regressor_.named_steps["mlp"]
    convergence = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
    for item in captured:
        if not issubclass(item.category, ConvergenceWarning):
            warnings.warn(str(item.message), item.category)
    status = {
        "FittedEpochs": int(network.n_iter_),
        "InputFeatures": int(x.shape[1]),
        "TrainableParameters": int(sum(a.size for a in network.coefs_ + network.intercepts_)),
        "ConvergenceWarning": bool(convergence), "ConvergenceMessages": convergence,
    }
    if convergence:
        print("MLP reached its fixed optimization limit; the convergence warning is recorded.", flush=True)
    return model, status


def save_enhanced_data(frame: pd.DataFrame, x: pd.DataFrame, split: str,
                       destination: Path, save_full: bool) -> None:
    """Export previews or optional full enhanced datasets, preserving row alignment."""
    metadata = frame[KEYS + [REFERENCE_TIME, FORECAST_TIME, TARGET]].reset_index(drop=True)
    preview = pd.concat([metadata.head(500), x.head(500).reset_index(drop=True)], axis=1)
    preview.to_csv(destination / f"week10_{split.lower()}_enhanced_sample.csv", index=False)
    if save_full:
        complete = pd.concat([metadata, x.reset_index(drop=True)], axis=1)
        complete.to_csv(destination / f"week10_{split.lower()}_enhanced_full.csv.gz", index=False, compression="gzip")


# ---------------------------------------------------------------------------
# 4. MODEL PERFORMANCE -- TRAINING, VALIDATION, AND TEST DATASETS
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 5. CONTROLLED DATA ENHANCEMENT EXPERIMENTS AND PRE-TEST MODEL SELECTION
# ---------------------------------------------------------------------------
def stages_of_features(all_names: list[str]) -> dict[str, list[str]]:
    """One reference plus THREE nested data interventions; fixed MLP_B settings."""
    history = [f"{key}{suffix}" for key in HISTORY_LOGS for suffix in ("Log1p", "Missing")]
    history += ["MonthlyLogChange", "MonthlyLogChangeMissing", "AnnualLogChange", "AnnualLogChangeMissing"]
    quality = [f"{key}{suffix}" for key in ORIGINAL_LOGS for suffix in ("Log1p", "Missing")]
    quality += ["LoadFactorClean", "LoadFactorCleanMissing", "ScheduleCompletionRate", "ScheduleCompletionRateMissing"]
    context = ["ForecastMonthSin", "ForecastMonthCos", "ForecastDaysRatio", "TrendMonth"]
    context += [f"{field}TrainingFrequency" for field in (*KEYS, "Route")]
    result = {BASELINE_NAME: ["PC1", "PC2"],
              STAGE_NAMES[0]: ["PC1", "PC2", *history],
              STAGE_NAMES[1]: ["PC1", "PC2", *history, *quality],
              STAGE_NAMES[2]: ["PC1", "PC2", *history, *quality, *context]}
    if any(len(cols) != len(set(cols)) or not set(cols).issubset(all_names) for cols in result.values()):
        raise ValueError("Invalid feature stages or column-order contract")
    assert list(result[STAGE_NAMES[0]]) == result[STAGE_NAMES[2]][:len(result[STAGE_NAMES[0]])]
    return result


def _reference_manifest(root: Path) -> dict:
    manifest_path = root / "Models" / "week9" / "week9_final_model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Week 9 final-model manifest: {manifest_path}. Copy the verified Week 9 manifest into Models/week9/.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("SelectedModel") != "MLP_B" or not manifest.get("ValidationConsistencyCheckPassed"):
        raise RuntimeError("Week 9 manifest must verify MLP_B and validation compatibility")
    if not manifest.get("TestEvaluatedAfterSelectionLock") or manifest.get("TestLabelsUsedToSelectModel"):
        raise RuntimeError("Week 9 final test safeguards are not documented")
    if manifest.get("SelectedModelArtifact") != BASELINE_ARTIFACT:
        raise RuntimeError("Unexpected original model artifact path; inspect Week 9 manifest")
    return manifest


def load_locked_baseline(root: Path, manifest: dict, week8) -> Any:
    filename = root / manifest["SelectedModelArtifact"]
    if not filename.is_file():
        raise FileNotFoundError(f"Saved Week 9-selected MLP_B model is missing: {filename}. Do not replace it with a freshly fitted baseline.")
    # Joblib is trusted only for project artifacts created locally by the user.
    if manifest.get("FrozenModelSha256"):
        digest = hashlib.sha256(filename.read_bytes()).hexdigest()
        if digest != manifest["FrozenModelSha256"]:
            raise RuntimeError("Saved Week 8 model SHA-256 differs from Week 9 manifest. Stop for provenance audit.")
    estimator = joblib.load(filename)
    cfg = estimator.regressor_.named_steps["mlp"].get_params()
    if tuple(cfg["hidden_layer_sizes"]) != HIDDEN_LAYERS or cfg["early_stopping"]:
        raise ValueError("Saved original model is not the locked Week 8 MLP_B configuration")
    for key, value in week8.COMMON.items():
        if cfg[key] != value:
            raise ValueError(f"Saved MLP_B setting '{key}' differs from expected Week 8 settings")
    return estimator


def _score_record(name: str, split: str, truth: np.ndarray, pred: np.ndarray, nparams: int, nfeatures: int):
    return {"Model": name, "Split": split, "Rows": len(truth), "InputFeatures": nfeatures,
            "TrainableParameters": nparams, **regression_metrics(truth, pred)}


def _all_stage_figures(metrics: pd.DataFrame, features: dict, original_month: pd.DataFrame,
                       refined_month: pd.DataFrame, figures: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 4.2))
    xs = np.arange(4)
    names = [BASELINE_NAME, *STAGE_NAMES]
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        for split in ("Training", "Validation"):
            values = metrics.query("Split == @split").set_index("Model").loc[names, metric]
            ax.plot(xs, values, marker="o", label=split)
        ax.set_xticks(xs, ["Original", "1. Temporal", "2. Quality", "3. Context"], rotation=15)
        ax.set_ylabel(f"{metric} (passengers)")
        ax.set_title(f"Week 10: {metric} across data enhancements")
        ax.legend()
    fig.tight_layout();fig.savefig(figures / "week10_validation_ablation_comparison.png", dpi=165);plt.close(fig)
    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    for table, label in ((original_month, "Original MLP_B"), (refined_month, "Full-enhancement candidate")):
        ax.plot(pd.to_datetime(table[FORECAST_TIME]), table.MAE, label=label)
    ax.set(xlabel="Training target month", ylabel="MAE (passengers)", title="Training error by forecast month")
    ax.legend();fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(figures / "week10_training_monthly_mae_before_after.png", dpi=165);plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ns = [len(features[x]) for x in names]
    ax.bar(["Original", "Temporal", "Quality", "Context"], ns)
    ax.set(ylabel="Number of predictors", title="Input-feature count across fixed-architecture models")
    fig.tight_layout();fig.savefig(figures / "week10_feature_dimension_stages.png", dpi=165);plt.close(fig)


def _test_visuals(test_frame: pd.DataFrame, truth: np.ndarray, pred: np.ndarray, figures: Path) -> pd.DataFrame:
    pred = np.maximum(np.asarray(pred, dtype=float), 0.0)
    selection = np.sort(np.random.default_rng(RANDOM_SEED).choice(len(truth), min(4000, len(truth)), replace=False))
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    y, yhat = truth[selection], pred[selection]
    ax.scatter(y, yhat, alpha=.25, s=8)
    cap = max(float(np.max(y)), float(np.max(yhat)), 1)
    ax.plot([0, cap], [0, cap], "--", label="Perfect prediction")
    ax.set(xlabel="Observed test passengers", ylabel="Predicted test passengers",
           title="Week 10 selected enhanced model: test display sample")
    ax.legend();fig.tight_layout()
    fig.savefig(figures / "week10_selected_test_actual_vs_predicted.png", dpi=170);plt.close(fig)
    monthly = pd.DataFrame({FORECAST_TIME: pd.to_datetime(test_frame[FORECAST_TIME]).to_numpy(),
                            "AbsoluteError": np.abs(truth - pred)}).groupby(FORECAST_TIME, as_index=False).agg(
                            MAE=("AbsoluteError", "mean"), Rows=("AbsoluteError", "size"))
    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    ax.plot(monthly[FORECAST_TIME], monthly["MAE"], marker="o")
    ax.set(xlabel="Test target month", ylabel="MAE (passengers)", title="Week 10 selected enhanced model: test MAE by month")
    fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(figures / "week10_selected_test_monthly_mae.png", dpi=170);plt.close(fig)
    return monthly


def run_week10(root: Path, save_full_data: bool = False, threads: int = 4) -> pd.DataFrame:
    """Perform full, validation-locked, historical-test-reuse evaluation.

    A completed result is cached: repeated notebook execution never rescores
    the already studied test partition. To change the protocol, start a new
    versioned experiment rather than delete evidence of a completed evaluation.
    """
    root = root.resolve()
    report_dir = root / "Reports" / "week10"
    tables = report_dir / "week10_tables"
    figures = report_dir / "week10_figures"
    model_dir = root / "Models" / "week10"
    data_dir = root / "Data" / "processed" / "week10"
    for folder in (tables, figures, model_dir, data_dir):
        folder.mkdir(parents=True, exist_ok=True)
    completed = tables / "week10_final_test_result.json"
    manifest_file = model_dir / "week10_final_model_manifest.json"
    if completed.exists() or manifest_file.exists():
        if not completed.exists() or not manifest_file.exists():
            raise RuntimeError("Partial prior final evaluation found. Reconcile manually before rerunning.")
        saved = json.loads(completed.read_text(encoding="utf-8"))
        locked = json.loads((tables / "week10_selection_lock.json").read_text(encoding="utf-8"))
        if saved.get("SelectedModel") != locked.get("Winner"):
            raise RuntimeError("Existing cached final evaluation and winner lock disagree")
        saved_model = json.loads(manifest_file.read_text(encoding="utf-8"))
        if saved_model.get("SelectedModel") != saved["SelectedModel"]:
            raise RuntimeError("Final model manifest differs from cached evaluation")
        print("Existing Week 10 final evaluation detected. Using saved tables; NOT scoring test again.")
        return pd.read_csv(tables / "week10_train_validation_test_metrics.csv")

    week5 = load_source(root / "Src" / "week5_src_features_dimensionality.py", "week10_prior_week5")
    week8 = load_source(root / "Src" / "week8_src_mlp_neural_network.py", "week10_prior_week8")
    prev = _reference_manifest(root)
    lock9 = root / "Reports" / "week9" / "week9_tables" / "week9_selection_lock.json"
    if not lock9.exists() or json.loads(lock9.read_text(encoding="utf-8")).get("Winner") != "MLP_B":
        raise RuntimeError("Missing or inconsistent Week 9 model-selection lock")
    baseline = load_locked_baseline(root, prev, week8)
    print("Loading full BTS monthly observations and unchanged Week 3 chronological partitions...", flush=True)
    monthly = week5.load_monthly_without_rewriting_reports(root)
    monthly = week5.create_cross_field_features(monthly)
    split_data, split_summary = week5.get_week3_splits(monthly, root)
    assert_split_integrity(split_data, split_summary)
    observed_counts = dict(zip(split_summary.Split.astype(str), split_summary.Rows.astype(int)))
    if observed_counts != REFERENCE_EXPECTED_COUNTS:
        raise RuntimeError(f"Week 3 split drift: {observed_counts} != {REFERENCE_EXPECTED_COUNTS}")
    original_signatures = split_signatures(split_data)
    split_summary.to_csv(tables / "week10_split_summary.csv", index=False)

    original_state = fit_original_pca(split_data["Training"], week5)
    train = split_data["Training"]
    x_old_train = original_pca_inputs(train, original_state, week5)
    y_train = train[TARGET].to_numpy(dtype="float64")
    rows, models, fit_details, training_monthly_errors = [], {BASELINE_NAME: baseline}, {}, {}
    orig_params = int(sum(x.size for x in baseline.regressor_.named_steps["mlp"].coefs_ + baseline.regressor_.named_steps["mlp"].intercepts_))
    with threadpool_limits(limits=threads):
        p = baseline.predict(x_old_train)
        rows.append(_score_record(BASELINE_NAME, "Training", y_train, p, orig_params, 2))
        training_monthly_errors[BASELINE_NAME] = training_error_analysis(train, p, BASELINE_NAME, tables, figures)
        old_xval = original_pca_inputs(split_data["Validation"], original_state, week5)
        valid_y = split_data["Validation"][TARGET].to_numpy(dtype="float64")
        old_val_score = regression_metrics(valid_y, baseline.predict(old_xval))
        if abs(old_val_score["MAE"] - prev["RecomputedValidationMAE"]) > .10 or abs(old_val_score["RMSE"] - prev["RecomputedValidationRMSE"]) > .10:
            raise RuntimeError("Saved Week 9 model / Week 5 PCA cannot reproduce historical validation metrics. STOP before enhancements/test.")
        print(f"Week 9 baseline validation compatibility passed: MAE {old_val_score['MAE']:,.2f}; RMSE {old_val_score['RMSE']:,.2f}")
        rows.append(_score_record(BASELINE_NAME, "Validation", valid_y, baseline.predict(old_xval), orig_params, 2))

        # Exact calendar history must be created before filtering supervised target rows.
        history = add_history_features(monthly)
        del monthly
        enhanced, enhanced_summary = week5.get_week3_splits(history, root)
        del history
        assert_split_integrity(enhanced, enhanced_summary)
        if not enhanced_summary.equals(split_summary) or split_signatures(enhanced) != original_signatures:
            raise RuntimeError("Enhancement changed records, target values, split boundaries or order")
        train_new = enhanced["Training"]
        fit_state = fit_enhancement_state(train_new, x_old_train)
        new_x_train, missing_train = transform_enhanced(train_new, x_old_train, fit_state)
        feature_groups = stages_of_features(fit_state["features"])
        missing_audits = [missing_train.assign(Split="Training")]
        new_x_valid, missing_valid = transform_enhanced(enhanced["Validation"], old_xval, fit_state)
        missing_audits.append(missing_valid.assign(Split="Validation"))
        fit_details[BASELINE_NAME] = {"FittedEpochs":int(baseline.regressor_.named_steps["mlp"].n_iter_),
                                       "TrainableParameters":orig_params,"Reference":"Week 9 locked estimator; not refitted"}
        original_monthly = pd.read_csv(tables / f"week10_{BASELINE_NAME}_training_errors_by_{FORECAST_TIME}.csv")
        stage_months = {}
        stage_status = {}
        for i, name in enumerate(STAGE_NAMES):
            cols = feature_groups[name]
            xtr = new_x_train[cols].to_numpy(dtype="float64")
            xva = new_x_valid[cols].to_numpy(dtype="float64")
            print(f"Training {name}: {len(cols)} features across {len(xtr):,} training records...", flush=True)
            fitted, status = fit_fixed_mlp(week8, xtr, y_train)
            stage_status[name] = status
            models[name] = fitted
            pred_train = fitted.predict(xtr)
            rows.append(_score_record(name, "Training", y_train, pred_train,
                                      status["TrainableParameters"], len(cols)))
            pred_val = fitted.predict(xva)
            rows.append(_score_record(name, "Validation", valid_y, pred_val,
                                      status["TrainableParameters"], len(cols)))
            print(f"  Validation MAE {rows[-1]['MAE']:,.2f}; RMSE {rows[-1]['RMSE']:,.2f}; epochs {status['FittedEpochs']}", flush=True)
            if i == 2:
                training_error_analysis(train_new, pred_train, name, tables, figures)
                stage_months[name] = pd.read_csv(tables / f"week10_{name}_training_errors_by_{FORECAST_TIME}.csv")
        save_enhanced_data(enhanced["Training"], new_x_train, "Training", data_dir, save_full_data)
        save_enhanced_data(enhanced["Validation"], new_x_valid, "Validation", data_dir, save_full_data)
        pd.concat(missing_audits, ignore_index=True).to_csv(tables / "week10_missingness_by_split.csv", index=False)
        pd.DataFrame({"Feature":fit_state["features"], "TrainingMedianImputation":fit_state["medians"].to_numpy()}).to_csv(
            tables / "week10_feature_manifest.csv", index=False)
        metrics = pd.DataFrame(rows)
        val = metrics.loc[metrics.Split.eq("Validation")].copy()
        val["PredeclaredStageOrder"] = val.Model.map({name: i for i,name in enumerate((BASELINE_NAME,*STAGE_NAMES))})
        val = val.sort_values(["MAE", "RMSE", "TrainableParameters", "PredeclaredStageOrder"],kind="mergesort")
        val.to_csv(tables / "week10_validation_ranking.csv", index=False)
        winner = str(val.iloc[0].Model)
        # Stage feature counts differ by design; architecture/optimizer/hyperparams unchanged.
        choice = {"Winner":winner,"ComparisonRule":"Lowest validation MAE; RMSE; trainable parameter count; predefined order",
                  "ValidationScores":{r.Model:{"MAE":float(r.MAE),"RMSE":float(r.RMSE)} for r in val.itertuples()},
                  "OriginalReference":"Locked Week 9 MLP_B", "TestEvaluatedAtSelection":False,
                  "Week9TestPreviouslyExamined":True}
        lockfile = tables / "week10_selection_lock.json"
        if lockfile.exists():
            previous = json.loads(lockfile.read_text(encoding="utf-8"))
            if previous.get("Winner") != winner:
                raise RuntimeError("Existing Week 10 selection lock differs; stop before accessing test")
            for name in previous.get("ValidationScores", {}):
                if abs(previous["ValidationScores"][name]["MAE"] - choice["ValidationScores"][name]["MAE"]) > .10:
                    raise RuntimeError("Week 10 validation metrics changed beyond tolerance; stop before accessing test")
        else:
            lockfile.write_text(json.dumps(choice, indent=2), encoding="utf-8")
        print(f"Selection LOCKED from validation only: {winner}", flush=True)
        metrics.to_csv(tables / "week10_train_validation_test_metrics.csv", index=False)
        _all_stage_figures(metrics, feature_groups, original_monthly, stage_months[STAGE_NAMES[-1]], figures)
        # Persist all fitted enhanced stages; final selection can legitimately be original MLP_B.
        # Safe saved bundle contains the explicitly frozen preprocessing; no test-fitted state.
        selected_model = models[winner]
        selected_cols = feature_groups[winner]
        if winner != BASELINE_NAME:
            fitted_bundle = {"model":selected_model,"original_pca":original_state,"enhancement":fit_state,
                             "feature_order":selected_cols,"target":TARGET,"reference_month_features_available_with_reporting_lag":True}
            saved_path = model_dir / "week10_selected_enhanced_mlp_bundle.joblib"
            joblib.dump(fitted_bundle, saved_path)
            loaded = joblib.load(saved_path)
            if not np.allclose(loaded["model"].predict(new_x_train[selected_cols].to_numpy()[:12]),
                               selected_model.predict(new_x_train[selected_cols].to_numpy()[:12])):
                raise RuntimeError("Saved selected model changed predictions after loading")
            del loaded
        else:
            saved_path = root / BASELINE_ARTIFACT
        # Freeze final candidate choice before any test target is scored in Week 10.
        test_frame = enhanced["Test"]
        # Data improvements apply to ALL splits, even if validation retains
        # the original model. No test labels are needed for this feature audit.
        test_pca = original_pca_inputs(test_frame, original_state, week5)
        x_test, missing_test = transform_enhanced(test_frame, test_pca, fit_state)
        missing_test.assign(Split="Test").to_csv(tables / "week10_test_missingness.csv", index=False)
        save_enhanced_data(test_frame, x_test, "Test", data_dir, save_full_data)
        if winner == BASELINE_NAME:
            # Reuse old independent historical test metrics, not re-score the already evaluated data.
            score = {"MAE":float(prev["FinalTestMAE"]),"RMSE":float(prev["FinalTestRMSE"]),"R2":float(prev["FinalTestR2"]),
                     "MeanResidual":None,"WAPE_Percent":None}
            test_mode = "Reuse unchanged Week 9 historical test metrics; no Week 10 test scoring"
        else:
            test_y = test_frame[TARGET].to_numpy(dtype="float64")
            selected_pred = selected_model.predict(x_test[selected_cols].to_numpy(dtype="float64"))
            score = regression_metrics(test_y, selected_pred)
            monthly_test = _test_visuals(test_frame, test_y, selected_pred, figures)
            monthly_test.to_csv(tables / "week10_selected_test_monthly_mae.csv", index=False)
            test_mode = "Week 10 selected enhanced model scored once on reused Week 9 test period"
            print(f"Week 10 selected model historical test MAE: {score['MAE']:,.2f}", flush=True)
        rows.append({"Model":winner,"Split":"Test","Rows":int(len(test_frame)),
                    "InputFeatures":len(selected_cols),"TrainableParameters":int(val.iloc[0].TrainableParameters), **score})
        final = pd.DataFrame(rows)
        final.to_csv(tables / "week10_train_validation_test_metrics.csv", index=False)
        # Required three-partition visualization uses only the final selection.
        chosen = final[final.Model.eq(winner)].set_index("Split").loc[list(SPLITS)]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
        for ax, metric in zip(axes, ("MAE", "RMSE")):
            ax.bar(list(SPLITS), chosen[metric])
            ax.set(ylabel=f"{metric} (passengers)", title=f"Selected model {metric}")
        fig.suptitle(f"Week 10 selected model: {winner}")
        fig.tight_layout()
        fig.savefig(figures / "week10_selected_train_validation_test.png", dpi=170)
        plt.close(fig)
        # Training-set error-band comparison: diagnosis, not held-out inference.
        def band_table(name: str):
            p = tables / f"week10_{name}_training_errors_by_DemandBand.csv"
            return pd.read_csv(p).set_index("DemandBand")["MAE"]
        before_bands = band_table(BASELINE_NAME)
        full_bands = band_table(STAGE_NAMES[-1])
        groups = list(before_bands.index)
        positions = np.arange(len(groups))
        fig, ax = plt.subplots(figsize=(9.5, 4.0))
        ax.bar(positions-.18, before_bands.reindex(groups), width=.36, label="Original MLP_B")
        ax.bar(positions+.18, full_bands.reindex(groups), width=.36, label="Fully enhanced MLP_B")
        ax.set_xticks(positions, groups, rotation=15)
        ax.set(ylabel="Training MAE (passengers)", title="Training errors by observed passenger-demand band")
        ax.legend();fig.tight_layout()
        fig.savefig(figures / "week10_training_demand_band_errors.png", dpi=170)
        plt.close(fig)
        wide = final.query('Split == "Validation"').set_index('Model')
        ablation = pd.DataFrame({"Stage": [BASELINE_NAME,*STAGE_NAMES],
                                "Validation_MAE": [float(wide.loc[n,"MAE"]) for n in [BASELINE_NAME,*STAGE_NAMES]],
                                "Validation_RMSE": [float(wide.loc[n,"RMSE"]) for n in [BASELINE_NAME,*STAGE_NAMES]],
                                "InputFeatures": [len(feature_groups[n]) for n in [BASELINE_NAME,*STAGE_NAMES]],
                                "TrainableParameters": [int(wide.loc[n,"TrainableParameters"]) for n in [BASELINE_NAME,*STAGE_NAMES]]})
        ablation["MAE_Improvement_vs_Original"] = ablation.iloc[0].Validation_MAE - ablation.Validation_MAE
        ablation["MAE_Improvement_vs_Previous"] = ablation.Validation_MAE.shift(1) - ablation.Validation_MAE
        ablation.to_csv(tables / "week10_incremental_ablation.csv", index=False)
        pd.DataFrame({"Stage": [BASELINE_NAME,*STAGE_NAMES],"AddedFeatures":["None", "Historical passenger lags and changes",
                      "Additional operational ratios and quality/missingness", "Forecast seasonality and training-derived route context"],
                      "FeatureCount":[len(feature_groups[n]) for n in [BASELINE_NAME,*STAGE_NAMES]]}).to_csv(
                      tables / "week10_three_data_improvements.csv",index=False)
        test_summary = {"SelectedModel":winner, "TrainingRows":int(len(y_train)),
                        "ValidationRows":int(len(valid_y)),"TestRows":int(len(test_frame)),
                        "TestMetrics":score, "TestMode":test_mode,"Week9TestPreviouslyExamined":True,
                        "TestLabelsUsedForSelection":False,"SelectionLock":str(lockfile.relative_to(root))}
        completed.write_text(json.dumps(test_summary, indent=2, allow_nan=False), encoding="utf-8")
        final_manifest = {"SelectedModel":winner,"SelectedModelPath":str(saved_path.relative_to(root)),
                          "OriginalModelUnchanged":True,"HiddenLayers":list(HIDDEN_LAYERS),
                          "InputFeatures":len(selected_cols),"TrainableParameters":int(val.iloc[0].TrainableParameters),
                          "FitScope":"Training only", "ValidationSelection":True,"TestUsedForModelSelection":False,
                          "Week9TestAlreadyViewed":True,"TestMetricsMode":test_mode,
                          "StageOrder":list(feature_groups), "StageColumns":feature_groups,
                          "FitStatus":stage_status,"Week8FixedHyperparameters":week8.COMMON,
                          "Warnings":"Historical monthly BTS inputs subject to publication delays; not verified as-of available"}
        manifest_file.write_text(json.dumps(final_manifest, indent=2, allow_nan=False), encoding="utf-8")
        summary = ["ADAN8888 Week 10 — verified results from local execution", "",
                   f"Selected on validation: {winner}",f"Train rows: {len(y_train):,}; validation rows: {len(valid_y):,}; test rows: {len(test_frame):,}",
                   "", "Training and validation comparisons:",
                   final.query('Split != "Test"').round(4).to_string(index=False),"", "Incremental data-improvement evidence:",
                   ablation.round(4).to_string(index=False), "", "Selected model historical test metrics:",
                   final.query('Split == "Test"').round(4).to_string(index=False), "",
                   "Test period already examined in Week 9; it is not a fresh independent holdout.",
                   "Data availability: BTS reference-month input publication lag not resolved for live deployment.",
                   "A business acceptance threshold has not been specified; operational readiness is undetermined."]
        (report_dir / "week10_verified_results_for_report.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
        print("Saved validation selection, ablation tables, selected historical test results and model manifest.")
        return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, help="Existing project root with Src and Data/raw")
    parser.add_argument("--save-full-data", action="store_true",help="Export every enhanced row (large CSV.gz outputs; not recommended for GitHub)")
    parser.add_argument("--threads",type=int,default=4,help="Number of BLAS/OpenMP worker threads (default 4)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    results = run_week10(detect_project_root(args.root),args.save_full_data,args.threads)
    if not results.empty:
        print(results[["Model","Split","Rows","MAE","RMSE","R2"]].round(3).to_string(index=False))
    print("All Week 10 checks passed.")
    print("===== WEEK 10 END =====")


if __name__ == "__main__":
    main()
