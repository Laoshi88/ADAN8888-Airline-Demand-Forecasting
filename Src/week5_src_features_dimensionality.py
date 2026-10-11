"""Week 5 — feature engineering and training-only PCA dimensionality reduction.

Extends Week 2/3 project.
Raw BTS T-100 archives remain unchanged; there is no synthetic data augmentation.

Run from the repository root:
    python Src/week5_src_features_dimensionality.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SERIES_KEYS = ["UniqueCarrier", "Origin", "Dest"]
REFERENCE_TIME = "Period"
FORECAST_TIME = "TargetPeriod"
TARGET = "TargetPassengersNextMonth"
BASE_FEATURES = ["Passengers", "Seats", "DepScheduled", "DepPerformed", "Distance"]
NEW_FEATURES = ["SeatsPerFlight", "PassengersPerFlight", "UnusedSeats", "PassengerMiles"]
CANDIDATE_FEATURES = BASE_FEATURES + NEW_FEATURES
PCA_COMPONENTS = 2
SPLIT_ORDER = ("Training", "Validation", "Test")


@dataclass
class Week5Results:
    split_summary: pd.DataFrame
    feature_definitions: pd.DataFrame
    feature_quality: pd.DataFrame
    dimension_summary: pd.DataFrame
    variance_summary: pd.DataFrame
    loadings: pd.DataFrame
    component_summary: pd.DataFrame
    split_quality: pd.DataFrame
    reduced_samples: dict[str, pd.DataFrame]
    medians: pd.Series
    kept_features: list[str]
    scaler: StandardScaler
    pca: PCA
    retained_variance_percent: float
    components_for_95pct: int
    full_variance_ratios: list[float]


def detect_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if all((folder / child).is_dir() for child in ("Src", "Notebooks", "Data")):
            return folder
    raise FileNotFoundError("Run from your repository root or its Notebooks/Src folder.")


def _import_source(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Required earlier-week source file is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # enables dataclasses in imported source
    spec.loader.exec_module(module)
    return module


def load_monthly_without_rewriting_reports(project_root: Path) -> pd.DataFrame:
    """Reuse Week 2 aggregation without its report-generation side effects."""
    week2 = _import_source(project_root / "Src" / "week2_src_t100_ingest_explore.py", "week5_week2")
    archives = week2.discover_archives(project_root / "Data" / "raw")
    monthly, _audit, _classes = week2.ingest_and_aggregate(archives)
    required = set(SERIES_KEYS + [REFERENCE_TIME, TARGET] + BASE_FEATURES)
    missing = required - set(monthly.columns)
    if missing:
        raise ValueError(f"Week 2 dataset is missing required fields: {sorted(missing)}")
    return monthly


def create_cross_field_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Four *current-month*, target-independent features, each joining two fields.

    A performed departure is a flight. Ratios with zero departures are undefined.
    Negative counts are missing; implausible negative unused seats are missing.
    Feature values are not allowed to contain infinity.
    """
    data = monthly.copy()
    for col in BASE_FEATURES:
        x = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        data[col] = x.mask(x.lt(0))

    data[REFERENCE_TIME] = pd.to_datetime(data[REFERENCE_TIME], errors="coerce")
    if data[REFERENCE_TIME].isna().any():
        raise ValueError("Every carrier-route-month observation needs a reference month.")
    if data.duplicated(SERIES_KEYS + [REFERENCE_TIME]).any():
        raise ValueError("Duplicated carrier-route-month keys: check Week 2 aggregation.")

    departures = data["DepPerformed"].where(data["DepPerformed"].gt(0))
    data["SeatsPerFlight"] = data["Seats"].div(departures)
    data["PassengersPerFlight"] = data["Passengers"].div(departures)
    unused = data["Seats"].sub(data["Passengers"])
    data["UnusedSeats"] = unused.mask(unused.lt(0))
    data["PassengerMiles"] = data["Passengers"].mul(data["Distance"])

    for col in NEW_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return data


def get_week3_splits(data: pd.DataFrame, root: Path):
    """Reuse Week 3's exact target eligibility and forecast-month split rules."""
    week3 = _import_source(root / "Src" / "week3_src_eda_split.py", "week5_week3")
    eligible, _eligibility_summary = week3.prepare_model_data(data)
    train, validation, test, _calendar, summary = week3.chronological_split(eligible)
    result = {"Training": train, "Validation": validation, "Test": test}
    for name, frame in result.items():
        if len(frame) == 0 or frame[TARGET].isna().any():
            raise AssertionError(f"{name} is empty or contains unknown target values")
    assert train[FORECAST_TIME].max() < validation[FORECAST_TIME].min()
    assert validation[FORECAST_TIME].max() < test[FORECAST_TIME].min()
    assert sum(len(v) for v in result.values()) == len(eligible)
    return result, summary


def _feature_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        ("SeatsPerFlight", "Seats / DepPerformed", "Average seats available per performed departure"),
        ("PassengersPerFlight", "Passengers / DepPerformed", "Average passengers carried per performed departure"),
        ("UnusedSeats", "Seats - Passengers, if nonnegative", "Unused reported capacity for the current month"),
        ("PassengerMiles", "Passengers * Distance", "Passenger-mile exposure combining current demand and distance"),
    ], columns=["Feature", "Formula", "Interpretation"])


def _numeric_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[CANDIDATE_FEATURES].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.mask(x.lt(0))
    # log1p is a deterministic, defined transformation of nonnegative values.
    return np.log1p(x)


def _visualize(results: Week5Results, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ratios = np.array(results.full_variance_ratios) * 100.0
    ax.bar(np.arange(1, len(ratios) + 1), ratios)
    ax.plot(np.arange(1, len(ratios) + 1), np.cumsum(ratios), marker="o", label="Cumulative explained variance")
    ax.axhline(95, linestyle="--", linewidth=1, label="95% variance reference")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Training variance explained (%)")
    ax.set_title("Week 5 PCA — Training Variance Retention")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "week5_figure_pca_explained_variance.png", dpi=175)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    values = results.loadings[["PC1", "PC2"]].to_numpy()
    im = ax.imshow(values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks([0, 1], labels=["PC1", "PC2"])
    ax.set_yticks(np.arange(len(results.loadings)), labels=results.loadings["Feature"])
    ax.set_title("Week 5 PCA Loadings — Training-Fitted Components")
    fig.colorbar(im, ax=ax, label="Component loading")
    fig.tight_layout()
    fig.savefig(figures_dir / "week5_figure_pca_loadings.png", dpi=175)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(["Initial inputs", "Varying inputs", "PC1 + PC2"], results.dimension_summary["Columns"])
    ax.set_ylabel("Number of model predictors")
    ax.set_title("Week 5 Dimensionality Reduction")
    fig.tight_layout()
    fig.savefig(figures_dir / "week5_figure_dimension_comparison.png", dpi=175)
    plt.close(fig)


def fit_feature_reduction(
    partitions: dict[str, pd.DataFrame],
    split_summary: pd.DataFrame,
    sample_rows: int = 500,
) -> Week5Results:
    """Use the training partition to fit imputation, standardization, and PCA.

    Validation and test features are projected with frozen training parameters only.
    Their TARGET column is metadata and NEVER an input to the decomposition.
    """
    if set(partitions) != set(SPLIT_ORDER):
        raise ValueError("Expected Training, Validation, and Test partitions.")
    train = partitions["Training"]
    numeric = {name: _numeric_inputs(partitions[name]) for name in SPLIT_ORDER}
    medians = numeric["Training"].median().fillna(0.0)
    # Train medians, rather than each split's median, are used for all splits.
    filled = {name: df.fillna(medians) for name, df in numeric.items()}
    kept_features = filled["Training"].columns[filled["Training"].nunique(dropna=False) > 1].tolist()
    if len(kept_features) <= PCA_COMPONENTS:
        raise ValueError("Need at least three nonconstant training predictors to reduce to two PCs.")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(filled["Training"][kept_features].to_numpy(dtype="float64"))
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full")
    pca.fit(x_train)
    full_pca = PCA(svd_solver="full")
    full_pca.fit(x_train)  # diagnostic only, training data only
    explained = full_pca.explained_variance_ratio_
    n_95 = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)

    reduced_samples: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for name in SPLIT_ORDER:
        source = partitions[name]
        transformed = pca.transform(scaler.transform(filled[name][kept_features].to_numpy(dtype="float64")))
        if not np.isfinite(transformed).all():
            raise ValueError(f"{name} has invalid PCA output")
        if len(source) != len(transformed):
            raise AssertionError(f"{name} changed row count")
        metadata = source[SERIES_KEYS + [REFERENCE_TIME, FORECAST_TIME, TARGET]].head(sample_rows).reset_index(drop=True)
        projected = pd.DataFrame(transformed[:sample_rows], columns=["PC1", "PC2"])
        reduced_samples[name] = pd.concat([metadata, projected], axis=1)
        quality_rows.append({"Split": name, "RowsBefore": len(source), "RowsAfter": len(transformed), "PCAColumns": 2,
                             "InvalidPCAValues": int((~np.isfinite(transformed)).sum())})
        for field in CANDIDATE_FEATURES:
            missing_rows.append({"Split": name, "Feature": field, "MissingBeforeImputation": int(numeric[name][field].isna().sum())})
        del transformed

    dimension_summary = pd.DataFrame([
        {"Step": "Original + cross-field numeric predictors", "Columns": len(CANDIDATE_FEATURES)},
        {"Step": "Nonconstant training predictors", "Columns": len(kept_features)},
        {"Step": "PCA reduced representation", "Columns": PCA_COMPONENTS},
    ])
    variance_summary = pd.DataFrame({"Component": np.arange(1, len(explained) + 1),
                                     "ExplainedVarianceRatio": explained,
                                     "CumulativeVarianceRatio": np.cumsum(explained)})
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"])
    loadings.insert(0, "Feature", kept_features)
    component_summary = pd.DataFrame([
        {"Metric": "Original numeric candidate predictors", "Value": len(CANDIDATE_FEATURES)},
        {"Metric": "Nonconstant training predictors", "Value": len(kept_features)},
        {"Metric": "PCA components retained", "Value": PCA_COMPONENTS},
        {"Metric": "Percent training variance captured by PC1+PC2", "Value": round(float(pca.explained_variance_ratio_.sum() * 100), 4)},
        {"Metric": "PCA components needed for 95% training variance", "Value": n_95},
        {"Metric": "Synthetic observations generated", "Value": 0},
    ])
    feature_quality = pd.DataFrame(missing_rows)
    return Week5Results(split_summary, _feature_definitions(), feature_quality,
                        dimension_summary, variance_summary, loadings, component_summary,
                        pd.DataFrame(quality_rows), reduced_samples, medians, kept_features, scaler, pca,
                        float(100 * pca.explained_variance_ratio_.sum()), n_95, explained.tolist())


def save_outputs(results: Week5Results, root: Path) -> None:
    tables = root / "Reports" / "week5" / "week5_tables"
    figures = root / "Reports" / "week5" / "week5_figures"
    samples = root / "Data" / "processed" / "week5"
    for folder in (tables, figures, samples):
        folder.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "split_summary": results.split_summary,
        "feature_definitions": results.feature_definitions,
        "missingness_audit": results.feature_quality,
        "dimension_summary": results.dimension_summary,
        "pca_explained_variance": results.variance_summary,
        "pca_loadings": results.loadings,
        "component_summary": results.component_summary,
        "split_quality_checks": results.split_quality,
    }.items():
        table.to_csv(tables / f"week5_{name}.csv", index=False)
    meta = {
        "input_features": CANDIDATE_FEATURES,
        "pca_input_features": results.kept_features,
        "log_transform": "log1p of nonnegative values",
        "training_medians_of_log_inputs": {k: float(v) for k, v in results.medians.items()},
        "training_scaler_mean": results.scaler.mean_.tolist(),
        "training_scaler_scale": results.scaler.scale_.tolist(),
        "pca_components": results.pca.components_.tolist(),
        "pca_training_center": results.pca.mean_.tolist(),
        "pca_explained_variance_ratios": results.pca.explained_variance_ratio_.tolist(),
        "variance_retained_two_components_percent": results.retained_variance_percent,
        "components_for_95_percent_training_variance": results.components_for_95pct,
        "fit_scope": "Training only",
        "augmentation": "None; no synthetic observations were added",
    }
    (tables / "week5_transform_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for name, table in results.reduced_samples.items():
        table.to_csv(samples / f"week5_{name.lower()}_reduced_sample.csv", index=False)
    (samples / "README.md").write_text(
        "# Week 5 processed-data samples\n\n"
        "Only representative PCA samples are committed to GitHub. The complete training, validation, "
        "and test frames are regenerated deterministically by Src/week5_src_features_dimensionality.py "
        "from unchanged BTS raw ZIPs and the prior-week Python source files.\n"
        "The full split CSVs are not uploaded because they would be unnecessarily large.\n", encoding="utf-8")
    _visualize(results, figures)


def run_week5(root: Path | None = None) -> Week5Results:
    root = root or detect_project_root()
    monthly = load_monthly_without_rewriting_reports(root)
    features = create_cross_field_features(monthly)
    partitions, split_summary = get_week3_splits(features, root)
    result = fit_feature_reduction(partitions, split_summary)
    save_outputs(result, root)
    return result


def print_summary(result: Week5Results) -> None:
    print("===== WEEK 5 START =====")
    print(result.split_summary[["Split", "Rows", "ForecastMonths", "FirstForecastMonth", "LastForecastMonth"]].to_string(index=False))
    print("\nCreated cross-field features:", ", ".join(NEW_FEATURES))
    print(result.component_summary.to_string(index=False))
    print("\nSplit integrity and finite PCA values:")
    print(result.split_quality.to_string(index=False))
    print("\nSaved artifacts under Reports/week5 and Data/processed/week5")
    print("All Week 5 checks passed.")
    print("===== WEEK 5 END =====")


if __name__ == "__main__":
    print_summary(run_week5())
