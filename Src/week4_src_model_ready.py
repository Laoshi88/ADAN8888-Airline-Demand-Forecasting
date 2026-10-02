"""Week 4 model-ready preprocessing for the ADAN8888 airline-demand project.

Project: BTS T-100 Domestic Scheduled-passenger Demand Forecasting.

Week 4 objectives implemented here:
1. Rebuild Week 2 carrier-route-month analytical table without rewriting prior-week outputs.
2. Recreate Week 3 leakage-resistant chronological train/v alidation/test split.
3. Engineer past-only history & calendar features with exact month continuityy.
4. Fit all learned preprocessing parameters on training data ONLY.
5. Applly frozen preprocessing rules to data (training, validation, test).
6. Save audit tables, figures, preprocessing metadata, samples.

- Full processed matrices are created in memory but are not written to GitHub by default since
  ~1.5+ million rows can produce unnecessarily large repository artifacts. 
- Small representative samples, feature metadata, fitted parameters, and reproducibility code 
  saved instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TARGET = "TargetPassengersNextMonth"
TIME_COLUMN = "Period"
TARGET_TIME_COLUMN = "TargetPeriod"
SERIES_KEYS = ["UniqueCarrier", "Origin", "Dest"]
CATEGORICAL_COLUMNS = ["UniqueCarrier", "Origin", "Dest"]

BASE_NUMERIC_COLUMNS = [
    "Passengers",
    "Seats",
    "DepScheduled",
    "DepPerformed",
    "Distance",
    "LoadFactor",
    "PassengerLag1",
    "PassengerLag2",
    "PassengerRolling3Mean",
    "ScheduleCompletionRate",
]

LOG_SOURCE_COLUMNS = [
    "Passengers",
    "Seats",
    "DepScheduled",
    "DepPerformed",
    "PassengerLag1",
    "PassengerLag2",
    "PassengerRolling3Mean",
]

SCALED_SOURCE_COLUMNS = [
    "PassengersLog1p",
    "SeatsLog1p",
    "DepScheduledLog1p",
    "DepPerformedLog1p",
    "PassengerLag1Log1p",
    "PassengerLag2Log1p",
    "PassengerRolling3MeanLog1p",
    "Distance",
    "LoadFactor",
    "ScheduleCompletionRate",
    "TrendMonth",
]

INDICATOR_COLUMNS = [
    "LoadFactorMissing",
    "LoadFactorInvalid",
    "PassengerLag1Missing",
    "PassengerLag2Missing",
    "PassengerRolling3MeanMissing",
    "ScheduleCompletionRateMissing",
    "UniqueCarrierMissing",
    "OriginMissing",
    "DestMissing",
]

FREQUENCY_COLUMNS = [
    "UniqueCarrierFreq",
    "OriginFreq",
    "DestFreq",
]

CALENDAR_COLUMNS = ["TargetMonthSin", "TargetMonthCos"]


@dataclass
class Week4Preprocessor:
    numeric_medians: dict[str, float]
    category_frequencies: dict[str, dict[str, float]]
    scaler_mean: dict[str, float]
    scaler_scale: dict[str, float]
    trend_origin: pd.Timestamp


@dataclass
class Week4Results:
    train_model_ready: pd.DataFrame
    validation_model_ready: pd.DataFrame
    test_model_ready: pd.DataFrame
    split_summary: pd.DataFrame
    processing_summary: pd.DataFrame
    feature_manifest: pd.DataFrame
    missingness_before_after: pd.DataFrame
    unknown_category_audit: pd.DataFrame
    preprocessor_parameters: pd.DataFrame
    data_quality_checks: pd.DataFrame
    preprocessor: Week4Preprocessor


def detect_project_root(start: Path | None = None) -> Path:
    """Locate project root from repository root or a child folder."""
    start = (start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / "Data").exists() and (path / "Src").exists() and (path / "Notebooks").exists():
            return path
    raise FileNotFoundError(
        "Could not locate ADAN8888 project root. Run from repository root or a child folder."
    )


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def rebuild_monthly_without_rewriting_prior_reports(
    project_root: Path,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Rebuild Week 2 monthly table withou calling Week2 output writer."""
    week2_path = project_root / "Src" / "week2_src_t100_ingest_explore.py"
    if not week2_path.exists():
        raise FileNotFoundError(f"Week 2 source file not found: {week2_path}")
    week2 = _load_module(week2_path, "week2_ingestion_for_week4")

    archives = week2.discover_archives(project_root / "Data" / "raw")
    monthly, _, _ = week2.ingest_and_aggregate(archives, chunksize=chunksize)
    return monthly


def add_past_only_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Add exact-calendar lag/rolling features using current and earlier months only."""
    out = monthly.copy()
    out[TIME_COLUMN] = pd.to_datetime(out[TIME_COLUMN], errors="coerce")

    base = out[SERIES_KEYS + [TIME_COLUMN, "Passengers"]].copy()
    for lag in (1, 2):
        lagged = base.copy()
        lagged[TIME_COLUMN] = lagged[TIME_COLUMN] + pd.DateOffset(months=lag)
        lagged = lagged.rename(columns={"Passengers": f"PassengerLag{lag}"})
        out = out.merge(
            lagged[SERIES_KEYS + [TIME_COLUMN, f"PassengerLag{lag}"]],
            on=SERIES_KEYS + [TIME_COLUMN],
            how="left",
            validate="one_to_one",
        )

    exact_three_month_history = out[["PassengerLag1", "PassengerLag2"]].notna().all(axis=1)
    out["PassengerRolling3Mean"] = np.where(
        exact_three_month_history,
        out[["Passengers", "PassengerLag1", "PassengerLag2"]].mean(axis=1),
        np.nan,
    )

    out["ScheduleCompletionRate"] = np.where(
        out["DepScheduled"] > 0,
        out["DepPerformed"] / out["DepScheduled"],
        np.nan,
    )
    return out


def rebuild_week3_splits(
    monthly: pd.DataFrame,
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reuse Week 3 target-eligibility and chronological split logic without rewriting Week 3 outputs."""
    week3_path = project_root / "Src" / "week3_src_eda_split.py"
    if not week3_path.exists():
        raise FileNotFoundError(f"Week 3 source file not found: {week3_path}")
    week3 = _load_module(week3_path, "week3_split_for_week4")

    model_data, _ = week3.prepare_model_data(monthly)
    train_df, validation_df, test_df, _, split_summary = week3.chronological_split(model_data)
    return model_data, train_df, validation_df, test_df, split_summary


def _clean_deterministic(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic data-quality rules that do not learn from any holdout data."""
    out = frame.copy()

    # Invalid physical load factors are treated as missing, not silently capped or deleted.
    out["LoadFactorInvalid"] = (
        out["LoadFactor"].lt(0) | out["LoadFactor"].gt(1)
    ).fillna(False).astype("int8")
    out.loc[out["LoadFactorInvalid"].eq(1), "LoadFactor"] = np.nan

    # Negative counts/distances are invalid if encountered. Week 3 observed none in training,
    # but the same deterministic rule is applied to all partitions for defensive reproducibility.
    for col in ["Passengers", "Seats", "DepScheduled", "DepPerformed", "Distance"]:
        out.loc[out[col].lt(0).fillna(False), col] = np.nan

    out["LoadFactorMissing"] = out["LoadFactor"].isna().astype("int8")
    out["PassengerLag1Missing"] = out["PassengerLag1"].isna().astype("int8")
    out["PassengerLag2Missing"] = out["PassengerLag2"].isna().astype("int8")
    out["PassengerRolling3MeanMissing"] = out["PassengerRolling3Mean"].isna().astype("int8")
    out["ScheduleCompletionRateMissing"] = out["ScheduleCompletionRate"].isna().astype("int8")

    for col in CATEGORICAL_COLUMNS:
        out[f"{col}Missing"] = out[col].isna().astype("int8")
        out[col] = out[col].astype("string").fillna("__MISSING__")

    return out


def fit_preprocessor(train_df: pd.DataFrame) -> Week4Preprocessor:
    """Fit medians, category frequencies, and scaling parameters on TRAINING data only."""
    clean = _clean_deterministic(train_df)

    numeric_medians: dict[str, float] = {}
    for col in BASE_NUMERIC_COLUMNS:
        median = pd.to_numeric(clean[col], errors="coerce").median()
        if pd.isna(median):
            median = 0.0
        numeric_medians[col] = float(median)

    imputed = clean.copy()
    for col, median in numeric_medians.items():
        imputed[col] = pd.to_numeric(imputed[col], errors="coerce").fillna(median)

    for col in LOG_SOURCE_COLUMNS:
        imputed[f"{col}Log1p"] = np.log1p(imputed[col].clip(lower=0))

    trend_origin = pd.Timestamp(imputed[TARGET_TIME_COLUMN].min()).to_period("M").to_timestamp()
    target_period = pd.to_datetime(imputed[TARGET_TIME_COLUMN])
    imputed["TrendMonth"] = (
        (target_period.dt.year - trend_origin.year) * 12
        + (target_period.dt.month - trend_origin.month)
    ).astype(float)

    category_frequencies: dict[str, dict[str, float]] = {}
    for col in CATEGORICAL_COLUMNS:
        freq = imputed[col].value_counts(normalize=True, dropna=False)
        category_frequencies[col] = {str(k): float(v) for k, v in freq.items()}

    scaler = StandardScaler()
    scaler.fit(imputed[SCALED_SOURCE_COLUMNS])
    scaler_mean = {c: float(v) for c, v in zip(SCALED_SOURCE_COLUMNS, scaler.mean_)}
    scaler_scale = {c: float(v) for c, v in zip(SCALED_SOURCE_COLUMNS, scaler.scale_)}

    return Week4Preprocessor(
        numeric_medians=numeric_medians,
        category_frequencies=category_frequencies,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        trend_origin=trend_origin,
    )


def transform_split(
    frame: pd.DataFrame,
    preprocessor: Week4Preprocessor,
) -> pd.DataFrame:
    """Apply fozen training-fitted preprocessing rules to one chronological split."""
    clean = _clean_deterministic(frame)

    for col, median in preprocessor.numeric_medians.items():
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(median)

    for col in LOG_SOURCE_COLUMNS:
        clean[f"{col}Log1p"] = np.log1p(clean[col].clip(lower=0))

    target_period = pd.to_datetime(clean[TARGET_TIME_COLUMN])
    clean["TrendMonth"] = (
        (target_period.dt.year - preprocessor.trend_origin.year) * 12
        + (target_period.dt.month - preprocessor.trend_origin.month)
    ).astype(float)
    clean["TargetMonthSin"] = np.sin(2 * np.pi * target_period.dt.month / 12.0)
    clean["TargetMonthCos"] = np.cos(2 * np.pi * target_period.dt.month / 12.0)

    for col in CATEGORICAL_COLUMNS:
        mapping = preprocessor.category_frequencies[col]
        clean[f"{col}Freq"] = clean[col].map(mapping).fillna(0.0).astype(float)

    scaled = clean[SCALED_SOURCE_COLUMNS].copy()
    for col in SCALED_SOURCE_COLUMNS:
        scale = preprocessor.scaler_scale[col]
        if scale == 0:
            scale = 1.0
        clean[f"z_{col}"] = (scaled[col] - preprocessor.scaler_mean[col]) / scale

    clean["TargetLog1p"] = np.log1p(clean[TARGET].clip(lower=0))

    feature_columns = [f"z_{c}" for c in SCALED_SOURCE_COLUMNS]
    feature_columns += CALENDAR_COLUMNS + FREQUENCY_COLUMNS + INDICATOR_COLUMNS

    metadata_columns = [TIME_COLUMN, TARGET_TIME_COLUMN, *SERIES_KEYS]
    output_columns = metadata_columns + [TARGET, "TargetLog1p"] + feature_columns
    out = clean[output_columns].copy()

    # Stable compact dtypes reduce memory without changing meaning.
    for col in feature_columns + ["TargetLog1p"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    return out


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model-input columns, excluding identifiers, dates, and targets."""
    excluded = {TIME_COLUMN, TARGET_TIME_COLUMN, *SERIES_KEYS, TARGET, "TargetLog1p"}
    return [c for c in frame.columns if c not in excluded]


def _build_feature_manifest(model_ready: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for col in model_feature_columns(model_ready):
        if col.startswith("z_"):
            family = "Standardized numeric"
            treatment = "Training-median imputation where needed; optional log1p; StandardScaler fit on training only."
        elif col.endswith("Freq"):
            family = "Categorical encoding"
            treatment = "Training-only frequency encoding; unseen holdout category maps to 0."
        elif col.endswith("Missing") or col == "LoadFactorInvalid":
            family = "Missingness/data-quality indicator"
            treatment = "Binary indicator preserves information about missing/invalid source state."
        elif col.startswith("TargetMonth"):
            family = "Calendar seasonality"
            treatment = "Deterministic cyclical encoding of known forecast calendar month."
        else:
            family = "Other"
            treatment = "Model-ready numeric feature."
        rows.append({"Feature": col, "Family": family, "Treatment": treatment})
    return pd.DataFrame(rows)


def _missingness_audit(
    raw_splits: dict[str, pd.DataFrame],
    processed_splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    audit_cols = [
        "LoadFactor",
        "PassengerLag1",
        "PassengerLag2",
        "PassengerRolling3Mean",
        "ScheduleCompletionRate",
    ]
    for split, raw in raw_splits.items():
        proc = processed_splits[split]
        for col in audit_cols:
            rows.append(
                {
                    "Split": split,
                    "Variable": col,
                    "MissingBefore": int(raw[col].isna().sum()),
                    "MissingAfter": 0,
                }
            )
    return pd.DataFrame(rows)


def _unknown_category_audit(
    raw_splits: dict[str, pd.DataFrame],
    preprocessor: Week4Preprocessor,
) -> pd.DataFrame:
    rows = []
    for split, frame in raw_splits.items():
        clean = _clean_deterministic(frame)
        for col in CATEGORICAL_COLUMNS:
            known = set(preprocessor.category_frequencies[col])
            unknown = ~clean[col].astype(str).isin(known)
            rows.append(
                {
                    "Split": split,
                    "Variable": col,
                    "Rows": len(clean),
                    "UnknownCategoryRows": int(unknown.sum()),
                    "UnknownCategoryPercent": float(100 * unknown.mean()) if len(clean) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _parameter_table(preprocessor: Week4Preprocessor) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col, value in preprocessor.numeric_medians.items():
        rows.append({"ParameterType": "Training median", "Variable": col, "Value": value})
    for col in SCALED_SOURCE_COLUMNS:
        rows.append({"ParameterType": "Scaler mean", "Variable": col, "Value": preprocessor.scaler_mean[col]})
        rows.append({"ParameterType": "Scaler scale", "Variable": col, "Value": preprocessor.scaler_scale[col]})
    rows.append({"ParameterType": "Trend origin", "Variable": "TargetPeriod", "Value": preprocessor.trend_origin.strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)


def _processing_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Step": 1,
                "Processing": "Preserve Week 3 chronological split",
                "Implementation": "Same ordered forecast-month 70/15/15 split; no random shuffling.",
                "LeakageControl": "Future months remain isolated from training.",
            },
            {
                "Step": 2,
                "Processing": "Treat invalid/missing numeric values",
                "Implementation": "Out-of-range load factor and defensive negative-value checks become missing; numeric medians learned on training only.",
                "LeakageControl": "Validation/test values do not influence imputation statistics.",
            },
            {
                "Step": 3,
                "Processing": "Engineer exact-calendar history",
                "Implementation": "Passenger lag-1, lag-2, and 3-month rolling mean use exact prior months only; missing-history indicators are retained.",
                "LeakageControl": "No future values and no silent bridging across missing months.",
            },
            {
                "Step": 4,
                "Processing": "Transform skewed numeric variables",
                "Implementation": "log1p for demand/capacity/count features; raw target preserved and TargetLog1p supplied as an alternative target.",
                "LeakageControl": "Transform is deterministic and target is never used to construct predictors.",
            },
            {
                "Step": 5,
                "Processing": "Encode high-cardinality categoricals",
                "Implementation": "Carrier, origin, and destination are frequency-encoded using training frequencies; unseen holdout levels map to 0.",
                "LeakageControl": "Holdout category frequencies are never learned.",
            },
            {
                "Step": 6,
                "Processing": "Encode seasonality and trend",
                "Implementation": "Sine/cosine forecast-month features and a deterministic month trend are created.",
                "LeakageControl": "Uses calendar information known at forecast time.",
            },
            {
                "Step": 7,
                "Processing": "Standardize numeric model inputs",
                "Implementation": "StandardScaler parameters are fit on training only and frozen for validation/test.",
                "LeakageControl": "Future means/variances never enter training transformations.",
            },
        ]
    )


def _quality_checks(
    processed_splits: dict[str, pd.DataFrame],
    split_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    expected = {str(r["Split"]): int(r["Rows"]) for _, r in split_summary.iterrows()}
    for split, frame in processed_splits.items():
        features = model_feature_columns(frame)
        vals = frame[features].to_numpy(dtype="float64", copy=False)
        rows.extend(
            [
                {"Split": split, "Check": "Row count retained", "Passed": len(frame) == expected[split], "Value": len(frame)},
                {"Split": split, "Check": "No missing model features", "Passed": not frame[features].isna().any().any(), "Value": int(frame[features].isna().sum().sum())},
                {"Split": split, "Check": "No infinite model features", "Passed": bool(np.isfinite(vals).all()), "Value": int((~np.isfinite(vals)).sum())},
                {"Split": split, "Check": "Target remains observed", "Passed": frame[TARGET].notna().all(), "Value": int(frame[TARGET].isna().sum())},
            ]
        )
    return pd.DataFrame(rows)


def _save_figures(
    train_raw: pd.DataFrame,
    split_summary: pd.DataFrame,
    missingness_audit: pd.DataFrame,
    unknown_audit: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: transformation of the right-skewed training target.
    target = train_raw[TARGET].dropna().clip(lower=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log1p(target), bins=60)
    ax.set_title("Week 4 Training Target After log1p Transformation")
    ax.set_xlabel("log(1 + next-month passengers)")
    ax.set_ylabel("Training observations")
    fig.tight_layout()
    fig.savefig(figures_dir / "week4_figure_training_target_log1p.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: missing engineered inputs before imputation. After model-ready processing they are zero.
    plot = (
        missingness_audit.loc[missingness_audit["Split"].eq("Training")]
        .groupby("Variable", as_index=False)["MissingBefore"].max()
        .sort_values("MissingBefore")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot["Variable"], plot["MissingBefore"])
    ax.set_title("Training Missing Values Before Week 4 Imputation")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(figures_dir / "week4_figure_training_missing_before_imputation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: model-eligible split row counts are retained because Week 4 imputes rather than drops.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(split_summary["Split"].astype(str), split_summary["Rows"])
    ax.set_title("Week 4 Model-ready Row Retention by Split")
    ax.set_xlabel("Partition")
    ax.set_ylabel("Rows retained")
    fig.tight_layout()
    fig.savefig(figures_dir / "week4_figure_split_row_retention.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: unseen categories encountered when applying training-fitted mappings.
    plot = unknown_audit.loc[~unknown_audit["Split"].eq("Training")].copy()
    if len(plot):
        pivot = plot.pivot(index="Variable", columns="Split", values="UnknownCategoryPercent").fillna(0)
        ax = pivot.plot(kind="bar", figsize=(8, 5))
        ax.set_title("Unseen Categorical Levels in Holdout Partitions")
        ax.set_xlabel("Categorical predictor")
        ax.set_ylabel("Rows with unseen level (%)")
        ax.tick_params(axis="x", rotation=0)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(figures_dir / "week4_figure_holdout_unknown_categories.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def save_outputs(
    results: Week4Results,
    project_root: Path,
    sample_rows: int = 500,
) -> None:
    """SaveWeek 4 artifacts avoiding giant processed datasets in GitHub."""
    week4_reports = project_root / "Reports" / "week4"
    tables_dir = week4_reports / "week4_tables"
    figures_dir = week4_reports / "week4_figures"
    processed_dir = project_root / "Data" / "processed" / "week4"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    results.processing_summary.to_csv(tables_dir / "week4_processing_summary.csv", index=False)
    results.feature_manifest.to_csv(tables_dir / "week4_feature_manifest.csv", index=False)
    results.missingness_before_after.to_csv(tables_dir / "week4_missingness_audit.csv", index=False)
    results.unknown_category_audit.to_csv(tables_dir / "week4_unknown_category_audit.csv", index=False)
    results.preprocessor_parameters.to_csv(tables_dir / "week4_preprocessor_parameters.csv", index=False)
    results.data_quality_checks.to_csv(tables_dir / "week4_data_quality_checks.csv", index=False)
    results.split_summary.to_csv(tables_dir / "week4_split_summary.csv", index=False)

    # Full category-frequency maps are small enough to audit and reproduce.
    frequency_rows = []
    for variable, mapping in results.preprocessor.category_frequencies.items():
        for category, frequency in mapping.items():
            frequency_rows.append({"Variable": variable, "Category": category, "TrainingFrequency": frequency})
    pd.DataFrame(frequency_rows).to_csv(tables_dir / "week4_category_frequency_maps.csv", index=False)

    params = {
        "numeric_medians": results.preprocessor.numeric_medians,
        "scaler_mean": results.preprocessor.scaler_mean,
        "scaler_scale": results.preprocessor.scaler_scale,
        "trend_origin": results.preprocessor.trend_origin.strftime("%Y-%m-%d"),
        "model_features": model_feature_columns(results.train_model_ready),
    }
    (tables_dir / "week4_preprocessing_metadata.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )

    split_frames = {
        "Training": results.train_model_ready,
        "Validation": results.validation_model_ready,
        "Test": results.test_model_ready,
    }
    for split, frame in split_frames.items():
        filename = f"week4_{split.lower()}_model_ready_sample.csv"
        frame.head(sample_rows).to_csv(processed_dir / filename, index=False)

    readme = (
        "# Week 4 artifacts\n\n"
        "Week 4 pipeline makes complete model-ready training, validation, and test frames in memory.\n"
        "To keep the GitHub repository compact, folder stores representative samples only.\n"
        "The full processed splits are deterministically reproducibl from Data/raw using\n"
        "`Src/week4_src_model_ready.py` & the saved preprocessing metadata under Reports/week4/week4_tables/.\n"
    )
    (processed_dir / "README.md").write_text(readme, encoding="utf-8")

    # Figures are regenerated from actual Week 4 run data.
    # Raw training data are not retained in Week4Results, so figure creation occurs in run_week4().


def run_week4_from_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_summary: pd.DataFrame,
    project_root: Path | None = None,
    save: bool = False,
    figures_raw_train: pd.DataFrame | None = None,
) -> Week4Results:
    """Fit and apply Week 4 preprocessing to already-created chronological splits."""
    preprocessor = fit_preprocessor(train_df)

    processed = {
        "Training": transform_split(train_df, preprocessor),
        "Validation": transform_split(validation_df, preprocessor),
        "Test": transform_split(test_df, preprocessor),
    }
    raw_splits = {"Training": train_df, "Validation": validation_df, "Test": test_df}

    feature_manifest = _build_feature_manifest(processed["Training"])
    missingness = _missingness_audit(raw_splits, processed)
    unknown = _unknown_category_audit(raw_splits, preprocessor)
    parameters = _parameter_table(preprocessor)
    processing = _processing_summary()
    quality = _quality_checks(processed, split_summary)

    results = Week4Results(
        train_model_ready=processed["Training"],
        validation_model_ready=processed["Validation"],
        test_model_ready=processed["Test"],
        split_summary=split_summary.copy(),
        processing_summary=processing,
        feature_manifest=feature_manifest,
        missingness_before_after=missingness,
        unknown_category_audit=unknown,
        preprocessor_parameters=parameters,
        data_quality_checks=quality,
        preprocessor=preprocessor,
    )

    if save:
        if project_root is None:
            raise ValueError("project_root is required when save=True")
        save_outputs(results, project_root)
        _save_figures(
            figures_raw_train if figures_raw_train is not None else train_df,
            split_summary,
            missingness,
            unknown,
            project_root / "Reports" / "week4" / "week4_figures",
        )
    return results


def run_week4(
    project_root: Path | None = None,
    chunksize: int = 200_000,
    sample_rows: int = 500,
) -> Week4Results:
    """Run Week 4 preprocessing from the project's raw BTS archives."""
    root = project_root or detect_project_root()

    monthly = rebuild_monthly_without_rewriting_prior_reports(root, chunksize=chunksize)
    monthly = add_past_only_features(monthly)
    _, train_df, validation_df, test_df, split_summary = rebuild_week3_splits(monthly, root)

    results = run_week4_from_splits(
        train_df,
        validation_df,
        test_df,
        split_summary,
        project_root=root,
        save=False,
    )
    save_outputs(results, root, sample_rows=sample_rows)
    _save_figures(
        train_df,
        split_summary,
        results.missingness_before_after,
        results.unknown_category_audit,
        root / "Reports" / "week4" / "week4_figures",
    )
    return results


def print_compact_summary(results: Week4Results) -> None:
    print("\n===== WEEK 4 START =====")
    print("\nChronological split retained from Week 3:")
    print(results.split_summary[["Split", "Rows", "ForecastMonths", "FirstForecastMonth", "LastForecastMonth"]].to_string(index=False))
    print(f"\nModel-ready feature count: {len(model_feature_columns(results.train_model_ready))}")
    print("\nQuality checks:")
    print(results.data_quality_checks.to_string(index=False))
    print("\nWeek 4 outputs saved under Reports/week4 and Data/processed/week4.")
    print("===== WEEK 4 END =====\n")


if __name__ == "__main__":
    results = run_week4()
    print_compact_summary(results)
