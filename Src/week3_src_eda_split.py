"""Week 3 Chronological Splitting And Training-only Eda For ADAN8888.

Project: BTS T-100 Domestic Scheduled-passenger Demand Forecasting.

Week 3 Objectives Implemented Here:
1. Rebuild Week 2 carrier-route-month analytical table from BTS ZIPs.
2. Create leakage-resistant 70%/15%/15% train/validation/test chronological split by
   *forecast month*, keeping every observation from same month in same partition.
3. Conduct exploratory data analysis (EDA) on TRAINING partition only.
4. Identify data-quality/modeling issues. Document preprocessing recommendations.
5. Save compact report-ready tables and figures.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import calendar
import importlib.util
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET = "TargetPassengersNextMonth"
TIME_COLUMN = "Period"
TARGET_TIME_COLUMN = "TargetPeriod"
SERIES_KEYS = ["UniqueCarrier", "Origin", "Dest"]
SPLIT_ORDER = ["Training", "Validation", "Test"]

TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15

EDA_NUMERIC = [
    "Passengers",
    "Seats",
    "DepScheduled",
    "DepPerformed",
    "Distance",
    "LoadFactor",
]
EDA_CATEGORICAL = ["UniqueCarrier", "Origin", "Dest", "Route"]


@dataclass
class Week3Results:
    monthly: pd.DataFrame
    model_data: pd.DataFrame
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    split_calendar: pd.DataFrame
    split_summary: pd.DataFrame
    eligibility_summary: pd.DataFrame
    training_profile: pd.DataFrame
    missingness: pd.DataFrame
    data_issues: pd.DataFrame
    numeric_summary: pd.DataFrame
    target_correlations: pd.DataFrame
    seasonality: pd.DataFrame
    monthly_trend: pd.DataFrame
    categorical_profile: pd.DataFrame
    key_findings: pd.DataFrame
    preprocessing_recommendations: pd.DataFrame


def detect_project_root(start: Path | None = None) -> Path:
    """Locate project root from repository root or a child folder."""
    start = (start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / "Data").exists() and (path / "Src").exists() and (path / "Notebooks").exists():
            return path
    raise FileNotFoundError(
        "Could not locate ADAN8888 project root. Run this script from repository "
        "root (or one of its child folders)."
    )


def load_week2_monthly(project_root: Path) -> pd.DataFrame:
    """Rebuild Week 2 carrier-route-month table using existing Week 2 module."""
    module_path = project_root / "Src" / "week2_src_t100_ingest_explore.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Required Week 2 source file was not found: {module_path}."
        )

    spec = importlib.util.spec_from_file_location("week2_ingestion", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Week 2 module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = module.run_week2(project_root)
    monthly = results.monthly.copy()

    required = set(SERIES_KEYS + [TIME_COLUMN, TARGET, "Passengers"])
    missing = required.difference(monthly.columns)
    if missing:
        raise ValueError(f"Week 2 monthly table is missing required columns: {sorted(missing)}")

    return monthly


def prepare_model_data(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create model-eligible frame and a pre-split eligibility audit.

    This is split preparation, not EDA. No validation/test distributions are inspected.
    Missing target values are not imputed because supervised outcome is unknown.
    """
    data = monthly.copy()
    data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN], errors="coerce")
    data[TARGET_TIME_COLUMN] = data[TIME_COLUMN] + pd.DateOffset(months=1)

    total_rows = len(data)
    missing_period = int(data[TIME_COLUMN].isna().sum())
    missing_target = int(data[TARGET].isna().sum())
    negative_target = int(data[TARGET].lt(0).fillna(False).sum())

    eligible_mask = (
        data[TIME_COLUMN].notna()
        & data[TARGET_TIME_COLUMN].notna()
        & data[TARGET].notna()
        & data[TARGET].ge(0)
    )
    model_data = data.loc[eligible_mask].copy()
    model_data = model_data.sort_values(
        [TARGET_TIME_COLUMN, *SERIES_KEYS]
    ).reset_index(drop=True)

    eligibility_summary = pd.DataFrame(
        {
            "Metric": [
                "Carrier-route-month rows before target eligibility",
                "Rows with missing reference month",
                "Rows with missing next-month target",
                "Rows with negative next-month target",
                "Rows eligible for Week 3 splitting/modeling",
                "Percent of rows eligible",
                "Percent missing next-month target",
            ],
            "Value": [
                total_rows,
                missing_period,
                missing_target,
                negative_target,
                len(model_data),
                round(100 * len(model_data) / total_rows, 4) if total_rows else np.nan,
                round(100 * missing_target / total_rows, 4) if total_rows else np.nan,
            ],
        }
    )
    return model_data, eligibility_summary


def chronological_split(
    model_data: pd.DataFrame,
    train_share: float = TRAIN_SHARE,
    validation_share: float = VALIDATION_SHARE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by ordered unique forecast months rather than randomly shuffled rows."""
    if not (0 < train_share < 1 and 0 < validation_share < 1):
        raise ValueError("train_share and validation_share must be between 0 and 1.")
    if train_share + validation_share >= 1:
        raise ValueError("train_share + validation_share must be less than 1.")

    available_months = pd.DatetimeIndex(
        sorted(pd.to_datetime(model_data[TARGET_TIME_COLUMN].dropna().unique()))
    )
    n_months = len(available_months)
    if n_months < 7:
        raise ValueError(
            "At least seven distinct forecast months are required for a meaningful 70/15/15 split."
        )

    n_train = int(np.floor(n_months * train_share))
    n_validation = int(np.floor(n_months * validation_share))
    n_test = n_months - n_train - n_validation

    # Defensive minimums for small data while preserving chronological order.
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError(
            f"Split produced an empty partition: train={n_train}, validation={n_validation}, test={n_test}."
        )

    train_months = available_months[:n_train]
    validation_months = available_months[n_train : n_train + n_validation]
    test_months = available_months[n_train + n_validation :]

    month_to_split: dict[pd.Timestamp, str] = {}
    month_to_split.update({pd.Timestamp(m): "Training" for m in train_months})
    month_to_split.update({pd.Timestamp(m): "Validation" for m in validation_months})
    month_to_split.update({pd.Timestamp(m): "Test" for m in test_months})

    split_calendar = pd.DataFrame(
        {
            TARGET_TIME_COLUMN: available_months,
            "Split": [month_to_split[pd.Timestamp(m)] for m in available_months],
        }
    )

    out = model_data.copy()
    out["Split"] = out[TARGET_TIME_COLUMN].map(month_to_split)
    out["Split"] = pd.Categorical(out["Split"], categories=SPLIT_ORDER, ordered=True)

    if out["Split"].isna().any():
        raise AssertionError("At least one eligible row was not assigned to a split.")

    train_df = out.loc[out["Split"] == "Training"].copy()
    validation_df = out.loc[out["Split"] == "Validation"].copy()
    test_df = out.loc[out["Split"] == "Test"].copy()

    # Integrity checks: no month may appear in more than one partition.
    assert train_df[TARGET_TIME_COLUMN].max() < validation_df[TARGET_TIME_COLUMN].min()
    assert validation_df[TARGET_TIME_COLUMN].max() < test_df[TARGET_TIME_COLUMN].min()
    assert set(train_df[TARGET_TIME_COLUMN]).isdisjoint(set(validation_df[TARGET_TIME_COLUMN]))
    assert set(train_df[TARGET_TIME_COLUMN]).isdisjoint(set(test_df[TARGET_TIME_COLUMN]))
    assert set(validation_df[TARGET_TIME_COLUMN]).isdisjoint(set(test_df[TARGET_TIME_COLUMN]))
    assert len(train_df) + len(validation_df) + len(test_df) == len(out)

    split_summary = (
        out.groupby("Split", observed=True)
        .agg(
            Rows=(TARGET, "size"),
            ForecastMonths=(TARGET_TIME_COLUMN, "nunique"),
            FirstReferenceMonth=(TIME_COLUMN, "min"),
            LastReferenceMonth=(TIME_COLUMN, "max"),
            FirstForecastMonth=(TARGET_TIME_COLUMN, "min"),
            LastForecastMonth=(TARGET_TIME_COLUMN, "max"),
        )
        .reindex(SPLIT_ORDER)
        .reset_index()
    )
    split_summary["RowPercent"] = 100 * split_summary["Rows"] / split_summary["Rows"].sum()
    split_summary["MonthPercent"] = (
        100 * split_summary["ForecastMonths"] / split_summary["ForecastMonths"].sum()
    )

    return train_df, validation_df, test_df, split_calendar, split_summary


def _training_profile(train_df: pd.DataFrame) -> pd.DataFrame:
    route_col = "Route" if "Route" in train_df.columns else None
    unique_routes = (
        int(train_df[route_col].nunique())
        if route_col
        else int(train_df[["Origin", "Dest"]].drop_duplicates().shape[0])
    )
    series_count = int(train_df[SERIES_KEYS].drop_duplicates().shape[0])
    return pd.DataFrame(
        {
            "Metric": [
                "Training rows",
                "Training columns",
                "First training forecast month",
                "Last training forecast month",
                "Training forecast months",
                "Unique carriers in training",
                "Unique directional routes in training",
                "Unique carrier-route series in training",
            ],
            "Value": [
                len(train_df),
                train_df.shape[1],
                train_df[TARGET_TIME_COLUMN].min(),
                train_df[TARGET_TIME_COLUMN].max(),
                train_df[TARGET_TIME_COLUMN].nunique(),
                train_df["UniqueCarrier"].nunique(),
                unique_routes,
                series_count,
            ],
        }
    )


def _training_missingness(train_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        TARGET,
        *EDA_NUMERIC,
        "UniqueCarrier",
        "Origin",
        "Dest",
    ]
    columns = [c for c in columns if c in train_df.columns]
    return (
        pd.DataFrame(
            {
                "Variable": columns,
                "MissingCount": [int(train_df[c].isna().sum()) for c in columns],
                "MissingPercent": [100 * train_df[c].isna().mean() for c in columns],
                "UniqueValues": [int(train_df[c].nunique(dropna=True)) for c in columns],
                "DataType": [str(train_df[c].dtype) for c in columns],
            }
        )
        .sort_values(["MissingPercent", "Variable"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _training_data_issues(train_df: pd.DataFrame) -> pd.DataFrame:
    """Quantify issues using training data only."""
    n = len(train_df)
    records: list[dict[str, Any]] = []

    def add(issue: str, count: int, severity: str, recommendation: str) -> None:
        records.append(
            {
                "Issue": issue,
                "Count": int(count),
                "PercentOfTrainingRows": (100 * int(count) / n) if n else np.nan,
                "Interpretation": severity,
                "RecommendedTreatment": recommendation,
            }
        )

    duplicate_keys = int(
        train_df[SERIES_KEYS + [TIME_COLUMN]].duplicated().sum()
    )
    add(
        "Duplicate carrier-route-month keys",
        duplicate_keys,
        "Data-integrity check",
        "Investigate duplicates before modeling; Week 2 aggregation should normally make this zero.",
    )

    for col in [TARGET, "Passengers", "Seats", "DepScheduled", "DepPerformed", "Distance"]:
        if col in train_df.columns:
            add(
                f"Negative values in {col}",
                int(train_df[col].lt(0).fillna(False).sum()),
                "Logically invalid if present",
                "Trace back to source and correct/exclude only demonstrably invalid records.",
            )

    if {"Seats", "Passengers"}.issubset(train_df.columns):
        add(
            "Positive passengers with zero/nonpositive seats",
            int(((train_df["Passengers"] > 0) & (train_df["Seats"] <= 0)).sum()),
            "Potential reporting inconsistency",
            "Investigate before modeling; do not silently impute a denominator.",
        )

    if "LoadFactor" in train_df.columns:
        add(
            "Load factor below 0",
            int(train_df["LoadFactor"].lt(0).fillna(False).sum()),
            "Logically invalid if present",
            "Investigate source records.",
        )
        add(
            "Load factor above 1",
            int(train_df["LoadFactor"].gt(1).fillna(False).sum()),
            "Potential aggregation/reporting inconsistency",
            "Review before deciding whether to cap, correct, or exclude; do not automatically delete.",
        )

    # Panel discontinuities are important for a next-month forecasting problem.
    ordered = train_df.sort_values(SERIES_KEYS + [TIME_COLUMN]).copy()
    month_index = ordered[TIME_COLUMN].dt.year * 12 + ordered[TIME_COLUMN].dt.month
    ordered["_month_index"] = month_index
    ordered["_gap"] = ordered.groupby(SERIES_KEYS, observed=True)["_month_index"].diff()
    gap_rows = int(ordered["_gap"].gt(1).sum())
    gap_series = int(
        ordered.loc[ordered["_gap"].gt(1), SERIES_KEYS].drop_duplicates().shape[0]
    )
    add(
        "Rows that follow a gap of more than one month within a carrier-route series",
        gap_rows,
        f"Unbalanced-panel continuity issue affecting {gap_series:,} training series",
        "Create lag/rolling features only when exact prior periods exist; preserve missing-history indicators.",
    )

    predictor_cols = [c for c in EDA_NUMERIC + ["UniqueCarrier", "Origin", "Dest"] if c in train_df.columns]
    missing_predictor_rows = int(train_df[predictor_cols].isna().any(axis=1).sum()) if predictor_cols else 0
    add(
        "Rows with at least one missing candidate predictor",
        missing_predictor_rows,
        "Missing-feature issue",
        "Fit any imputation rules on training data only and apply unchanged to validation/test later.",
    )

    return pd.DataFrame(records)


def _numeric_summary(train_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in [*EDA_NUMERIC, TARGET] if c in train_df.columns]
    if not cols:
        return pd.DataFrame()
    desc = train_df[cols].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    desc["median"] = train_df[cols].median()
    desc["skewness"] = train_df[cols].skew(numeric_only=True)
    desc["kurtosis"] = train_df[cols].kurt(numeric_only=True)
    desc = desc.reset_index().rename(columns={"index": "Variable"})
    return desc


def _target_correlations(train_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EDA_NUMERIC if c in train_df.columns]
    if not cols:
        return pd.DataFrame(columns=["Predictor", "PearsonCorrelationWithTarget", "AbsoluteCorrelation"])
    corr = train_df[cols + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
    out = (
        corr.rename("PearsonCorrelationWithTarget")
        .reset_index()
        .rename(columns={"index": "Predictor"})
    )
    out["AbsoluteCorrelation"] = out["PearsonCorrelationWithTarget"].abs()
    return out.sort_values("AbsoluteCorrelation", ascending=False).reset_index(drop=True)


def _seasonality(train_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        train_df.assign(ForecastCalendarMonth=train_df[TARGET_TIME_COLUMN].dt.month)
        .groupby("ForecastCalendarMonth", as_index=False)
        .agg(
            MeanTargetPassengers=(TARGET, "mean"),
            MedianTargetPassengers=(TARGET, "median"),
            TotalTargetPassengers=(TARGET, "sum"),
            Observations=(TARGET, "size"),
        )
    )
    out["MonthName"] = out["ForecastCalendarMonth"].map(
        lambda x: calendar.month_abbr[int(x)]
    )
    return out


def _monthly_trend(train_df: pd.DataFrame) -> pd.DataFrame:
    return (
        train_df.groupby(TARGET_TIME_COLUMN, as_index=False)
        .agg(
            TotalTargetPassengers=(TARGET, "sum"),
            MeanTargetPassengers=(TARGET, "mean"),
            MedianTargetPassengers=(TARGET, "median"),
            CarrierRouteObservations=(TARGET, "size"),
        )
        .sort_values(TARGET_TIME_COLUMN)
        .reset_index(drop=True)
    )


def _categorical_profile(train_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    data = train_df.copy()
    if "Route" not in data.columns and {"Origin", "Dest"}.issubset(data.columns):
        data["Route"] = data["Origin"].astype(str) + "→" + data["Dest"].astype(str)

    for col in [c for c in EDA_CATEGORICAL if c in data.columns]:
        counts = data[col].value_counts(dropna=False)
        top_value = counts.index[0] if len(counts) else None
        top_count = int(counts.iloc[0]) if len(counts) else 0
        records.append(
            {
                "Variable": col,
                "UniqueValues": int(data[col].nunique(dropna=True)),
                "MissingCount": int(data[col].isna().sum()),
                "MostFrequentValue": str(top_value),
                "MostFrequentCount": top_count,
                "MostFrequentPercent": 100 * top_count / len(data) if len(data) else np.nan,
            }
        )
    return pd.DataFrame(records)


def _preprocessing_recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Priority": 1,
                "Recommendation": "Preserve chronological month-based split and never randomly shuffle observations across time.",
                "Rationale": "project predicts future demand; temporal leakage would inflate validation/test performance.",
            },
            {
                "Priority": 2,
                "Recommendation": "Do not impute missing target values; exclude rows for which exact next-month outcome is unavailable.",
                "Rationale": "supervised outcome is unknown, not merely a missing predictor.",
            },
            {
                "Priority": 3,
                "Recommendation": "Fit all predictor imputation, encoding, scaling, and transformation rules on training data only.",
                "Rationale": "Using validation/test distributions to design preprocessing leaks information from future periods.",
            },
            {
                "Priority": 4,
                "Recommendation": "Represent carrier/origin/destination categorical information with a modeling-appropriate encoding learned from training data.",
                "Rationale": "These are high-cardinality identifiers and cannot be passed as raw strings to most regression estimators.",
            },
            {
                "Priority": 5,
                "Recommendation": "Consider log1p target modeling, robust losses, or tree-based models rather than deleting legitimate high-volume routes as outliers.",
                "Rationale": "Route-month passenger demand is expected to be strongly right-skewed and operational extremes may be genuine rather than errors.",
            },
            {
                "Priority": 6,
                "Recommendation": "Engineer lags/rolling features only from information available at or before forecast origin, with explicit gap handling.",
                "Rationale": "carrier-route panel is unbalanced; naive shift/rolling operations can silently bridge missing months.",
            },
            {
                "Priority": 7,
                "Recommendation": "Consider calendar seasonality and a pandemic/regime indicator during feature engineering/model comparison.",
                "Rationale": "time series spans materially different demand regimes and seasonal patterns.",
            },
        ]
    )


def _key_findings(
    monthly: pd.DataFrame,
    train_df: pd.DataFrame,
    correlations: pd.DataFrame,
    seasonality: pd.DataFrame,
    monthly_trend: pd.DataFrame,
    data_issues: pd.DataFrame,
) -> pd.DataFrame:
    missing_target = int(monthly[TARGET].isna().sum())
    missing_target_pct = 100 * missing_target / len(monthly) if len(monthly) else np.nan
    target_mean = float(train_df[TARGET].mean())
    target_median = float(train_df[TARGET].median())
    target_skew = float(train_df[TARGET].skew())

    if len(correlations):
        strongest = correlations.iloc[0]
        strongest_text = (
            f"{strongest['Predictor']} has largest absolute Pearson correlation with "
            f"next-month target ({strongest['PearsonCorrelationWithTarget']:.3f})."
        )
    else:
        strongest_text = "No numeric predictor correlations were available."

    if len(seasonality):
        peak = seasonality.loc[seasonality["MeanTargetPassengers"].idxmax()]
        low = seasonality.loc[seasonality["MeanTargetPassengers"].idxmin()]
        seasonal_text = (
            f"Average next-month demand is highest in {peak['MonthName']} "
            f"({peak['MeanTargetPassengers']:,.1f}) and lowest in {low['MonthName']} "
            f"({low['MeanTargetPassengers']:,.1f}) in training partition."
        )
    else:
        seasonal_text = "Seasonality could not be summarized."

    if len(monthly_trend):
        low_month = monthly_trend.loc[monthly_trend["TotalTargetPassengers"].idxmin()]
        regime_text = (
            f"lowest aggregate training demand occurs in "
            f"{pd.Timestamp(low_month[TARGET_TIME_COLUMN]).strftime('%Y-%m')}, "
            f"with {low_month['TotalTargetPassengers']:,.0f} target passengers, showing a material time-regime shock."
        )
    else:
        regime_text = "Time-trend information was unavailable."

    gap_row = data_issues.loc[
        data_issues["Issue"].str.startswith("Rows that follow a gap"), "Count"
    ]
    gap_count = int(gap_row.iloc[0]) if len(gap_row) else 0

    records = [
        {
            "Finding": "Structural target availability",
            "Evidence": (
                f"{missing_target:,} of {len(monthly):,} carrier-route-month rows "
                f"({missing_target_pct:.2f}%) do not have an exact observed next-month target. "
                "These rows are excluded from supervised modeling rather than imputed."
            ),
        },
        {
            "Finding": "Target distribution",
            "Evidence": (
                f"In training, target mean is {target_mean:,.2f}, median is "
                f"{target_median:,.2f}, and skewness is {target_skew:.2f}. "
                "mean/median gap and skewness quantify demand asymmetry."
            ),
        },
        {
            "Finding": "Strongest numeric relationship",
            "Evidence": strongest_text,
        },
        {
            "Finding": "Seasonality",
            "Evidence": seasonal_text,
        },
        {
            "Finding": "Time-regime variation",
            "Evidence": regime_text,
        },
        {
            "Finding": "Unbalanced carrier-route panel",
            "Evidence": (
                f"{gap_count:,} training observations follow a gap longer than one month in their "
                "carrier-route series, so future lag/rolling features must enforce exact calendar continuity."
            ),
        },
    ]
    return pd.DataFrame(records)


def build_training_eda(
    monthly: pd.DataFrame,
    train_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create all Week 3 EDA outputs using training partition only."""
    training_profile = _training_profile(train_df)
    missingness = _training_missingness(train_df)
    data_issues = _training_data_issues(train_df)
    numeric_summary = _numeric_summary(train_df)
    target_correlations = _target_correlations(train_df)
    seasonality = _seasonality(train_df)
    monthly_trend = _monthly_trend(train_df)
    categorical_profile = _categorical_profile(train_df)
    key_findings = _key_findings(
        monthly,
        train_df,
        target_correlations,
        seasonality,
        monthly_trend,
        data_issues,
    )
    return (
        training_profile,
        missingness,
        data_issues,
        numeric_summary,
        target_correlations,
        seasonality,
        monthly_trend,
        categorical_profile,
        key_findings,
    )


def _save_figures(
    train_df: pd.DataFrame,
    split_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    seasonality: pd.DataFrame,
    monthly_trend: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Split summary: partition sizes only (not exploratory analysis of future data).
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(split_summary["Split"].astype(str), split_summary["Rows"])
    ax.set_title("Chronological 70/15/15 Partition Sizes")
    ax.set_xlabel("Partition")
    ax.set_ylabel("Carrier-route-month rows")
    for i, row in split_summary.iterrows():
        ax.text(i, row["Rows"], f"{row['RowPercent']:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(figures_dir / "week3_figure_split_sizes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Training target distribution; log1p is used only for visualization.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log1p(train_df[TARGET].dropna()), bins=60)
    ax.set_title("Training Target Distribution")
    ax.set_xlabel("log(1 + next-month passengers)")
    ax.set_ylabel("Training observations")
    fig.tight_layout()
    fig.savefig(figures_dir / "week3_figure_training_target_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(monthly_trend[TARGET_TIME_COLUMN], monthly_trend["TotalTargetPassengers"])
    ax.set_title("Training Passenger Demand Over Time")
    ax.set_xlabel("Forecast month")
    ax.set_ylabel("Total next-month passengers")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures_dir / "week3_figure_training_demand_over_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(seasonality["MonthName"], seasonality["MeanTargetPassengers"])
    ax.set_title("Training Average Demand by Calendar Month")
    ax.set_xlabel("Forecast month")
    ax.set_ylabel("Mean next-month passengers")
    fig.tight_layout()
    fig.savefig(figures_dir / "week3_figure_training_seasonality.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if len(correlations):
        plot = correlations.sort_values("PearsonCorrelationWithTarget")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(plot["Predictor"], plot["PearsonCorrelationWithTarget"])
        ax.axvline(0, linewidth=0.8)
        ax.set_title("Training Numeric Predictor Correlation With Target")
        ax.set_xlabel("Pearson correlation")
        fig.tight_layout()
        fig.savefig(figures_dir / "week3_figure_training_target_correlations.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _iso(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _records_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        records.append({k: _iso(v) for k, v in record.items()})
    return records


def save_outputs(results: Week3Results, project_root: Path) -> None:
    reports_dir = project_root / "Reports"
    week3_reports_dir = reports_dir / "week3"
    tables_dir = week3_reports_dir / "week3_tables"
    figures_dir = week3_reports_dir / "week3_figures"

    week3_reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "week3_split_calendar.csv": results.split_calendar,
        "week3_split_summary.csv": results.split_summary,
        "week3_model_eligibility_summary.csv": results.eligibility_summary,
        "week3_training_profile.csv": results.training_profile,
        "week3_training_missingness.csv": results.missingness,
        "week3_training_data_issues.csv": results.data_issues,
        "week3_training_numeric_summary.csv": results.numeric_summary,
        "week3_training_target_correlations.csv": results.target_correlations,
        "week3_training_seasonality.csv": results.seasonality,
        "week3_training_monthly_trend.csv": results.monthly_trend,
        "week3_training_categorical_profile.csv": results.categorical_profile,
        "week3_key_findings.csv": results.key_findings,
        "week3_preprocessing_recommendations.csv": results.preprocessing_recommendations,
    }
    for name, df in tables.items():
        df.to_csv(tables_dir / name, index=False)

    _save_figures(
        results.train_df,
        results.split_summary,
        results.target_correlations,
        results.seasonality,
        results.monthly_trend,
        figures_dir,
    )

    summary = {
        "split_summary": _records_for_json(results.split_summary),
        "eligibility_summary": _records_for_json(results.eligibility_summary),
        "training_profile": _records_for_json(results.training_profile),
        "data_issues": _records_for_json(results.data_issues),
        "key_findings": _records_for_json(results.key_findings),
        "preprocessing_recommendations": _records_for_json(results.preprocessing_recommendations),
    }
    (tables_dir / "week3_results_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

def run_week3(project_root: Path | None = None) -> Week3Results:
    root = project_root or detect_project_root()
    monthly = load_week2_monthly(root)
    model_data, eligibility_summary = prepare_model_data(monthly)
    train_df, validation_df, test_df, split_calendar, split_summary = chronological_split(model_data)

    (
        training_profile,
        missingness,
        data_issues,
        numeric_summary,
        target_correlations,
        seasonality,
        monthly_trend,
        categorical_profile,
        key_findings,
    ) = build_training_eda(monthly, train_df)

    results = Week3Results(
        monthly=monthly,
        model_data=model_data,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        split_calendar=split_calendar,
        split_summary=split_summary,
        eligibility_summary=eligibility_summary,
        training_profile=training_profile,
        missingness=missingness,
        data_issues=data_issues,
        numeric_summary=numeric_summary,
        target_correlations=target_correlations,
        seasonality=seasonality,
        monthly_trend=monthly_trend,
        categorical_profile=categorical_profile,
        key_findings=key_findings,
        preprocessing_recommendations=_preprocessing_recommendations(),
    )
    save_outputs(results, root)
    return results


def print_compact_summary(results: Week3Results) -> None:
    print("\n===== WEEK 3 START =====")
    print("\nChronological split summary:")
    print(results.split_summary.to_string(index=False))
    print("\nModel eligibility summary:")
    print(results.eligibility_summary.to_string(index=False))
    print("\nTraining-only key findings:")
    print(results.key_findings.to_string(index=False))
    print("\nTraining-only data issues:")
    print(results.data_issues.to_string(index=False))
    print("\nWeek 3 outputs saved under Reports/week3/week3_tables and Reports/week3/week3_figures.")
    print("===== WEEK 3 END =====\n")


if __name__ == "__main__":
    results = run_week3()
    print_compact_summary(results)
