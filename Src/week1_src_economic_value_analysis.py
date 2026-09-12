"""Week 1 Economic-Value Analysis — Airline Demand Project.

Reproduce Quantitative Business-Case Calculations Used In
Week 1 Report. It Intentionally Does Not Analyze The BTS T-100 Observations;
That Data-ingestion And Exploratory Work Begins In Week 2.

From The Project Root:
    python .\\src\\week1_src_economic_value_analysis.py

Inputs
------
reports/week1_report_economic_value_assumptions.csv

Outputs
-------
reports/week1_report_economic_value_results.csv
reports/week1_report_figures/week1_report_economic_value_sensitivity.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ASSUMPTIONS = Path("reports/week1_report_economic_value_assumptions.csv")
DEFAULT_RESULTS = Path("reports/week1_report_economic_value_results.csv")
DEFAULT_FIGURE = Path(
    "reports/week1_report_figures/week1_report_economic_value_sensitivity.png"
)

REQUIRED_COLUMNS = {
    "scenario",
    "monthly_seats",
    "error_improvement_pp",
    "actionable_share",
    "value_per_actionable_seat",
}


def calculate_economic_value(assumptions: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly and annual gross decision value by scenario.

    Formula:
        better_aligned_seats_monthly = monthly_seats * error_improvement_pp / 100
        actionable_seats_monthly = better_aligned_seats_monthly * actionable_share
        monthly_gross_value = actionable_seats_monthly * value_per_actionable_seat
        annual_gross_value_calculated = monthly_gross_value * 12

    The calculations are scenario-based Week 1 business-case assumptions, not
    measured causal effects from the BTS dataset.
    """
    missing = REQUIRED_COLUMNS.difference(assumptions.columns)
    if missing:
        raise ValueError(f"Missing required assumption columns: {sorted(missing)}")

    results = assumptions.copy()
    results["better_aligned_seats_monthly"] = (
        results["monthly_seats"] * results["error_improvement_pp"] / 100.0
    )
    results["actionable_seats_monthly"] = (
        results["better_aligned_seats_monthly"] * results["actionable_share"]
    )
    results["monthly_gross_value_calculated"] = (
        results["actionable_seats_monthly"] * results["value_per_actionable_seat"]
    )
    results["annual_gross_value_calculated"] = (
        results["monthly_gross_value_calculated"] * 12.0
    )

    if "annual_gross_value" in results.columns:
        results["annual_value_difference"] = (
            results["annual_gross_value_calculated"] - results["annual_gross_value"]
        )
        results["annual_value_matches_report"] = (
            results["annual_value_difference"].abs() < 0.01
        )

    return results


def save_sensitivity_figure(results: pd.DataFrame, output_path: Path) -> None:
    """Save a simple annual gross decision-value sensitivity chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    values_millions = results["annual_gross_value_calculated"] / 1_000_000
    bars = ax.bar(results["scenario"], values_millions)
    ax.set_title("Illustrative Annual Gross Decision Value by Scenario")
    ax.set_ylabel("Annual gross decision value ($ millions)")
    ax.set_xlabel("Scenario")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values_millions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${value:,.2f}M",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Week 1 assumption-based economic-value analysis."
    )
    parser.add_argument(
        "--assumptions",
        type=Path,
        default=DEFAULT_ASSUMPTIONS,
        help=f"Assumptions CSV (default: {DEFAULT_ASSUMPTIONS})",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Detailed result CSV (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=DEFAULT_FIGURE,
        help=f"Sensitivity figure path (default: {DEFAULT_FIGURE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assumptions = pd.read_csv(args.assumptions)
    results = calculate_economic_value(assumptions)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results, index=False)
    save_sensitivity_figure(results, args.figure)

    display_columns = [
        "scenario",
        "better_aligned_seats_monthly",
        "actionable_seats_monthly",
        "monthly_gross_value_calculated",
        "annual_gross_value_calculated",
    ]
    print("Week 1 economic-value analysis complete.\n")
    print(results[display_columns].to_string(index=False))

    if "annual_value_matches_report" in results.columns:
        all_match = bool(results["annual_value_matches_report"].all())
        print(f"\nCalculated annual values match the report assumptions: {all_match}")

    print(f"Results written to: {args.results}")
    print(f"Figure written to:  {args.figure}")


if __name__ == "__main__":
    main()
