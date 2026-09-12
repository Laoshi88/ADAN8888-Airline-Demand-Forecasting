"""Utilities for inspecting and loading BTS T-100 annual ZIP extracts.

These helpers are intentionally small and reusable so notebooks stay focused on
analysis rather than file-handling boilerplate.
"""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import pandas as pd

EXPECTED_CORE_FIELDS = {
    "PASSENGERS",
    "SEATS",
    "DEPARTURES_SCHEDULED",
    "DEPARTURES_PERFORMED",
    "DISTANCE",
    "ORIGIN",
    "DEST",
    "YEAR",
    "MONTH",
}


def list_t100_zips(raw_dir: str | Path = "data/raw") -> list[Path]:
    """Return annual T-100 ZIP files in chronological filename order."""
    return sorted(Path(raw_dir).glob("T100_Domestic_Segment_All_Carriers_*.zip"))


def archive_members(zip_path: str | Path) -> list[str]:
    """List files stored inside one BTS ZIP archive."""
    with ZipFile(zip_path) as zf:
        return zf.namelist()


def read_t100_zip(zip_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the first CSV/TXT-like table inside a BTS ZIP into pandas."""
    zip_path = Path(zip_path)
    with ZipFile(zip_path) as zf:
        candidates = [
            name for name in zf.namelist()
            if name.lower().endswith((".csv", ".txt")) and not name.endswith("/")
        ]
        if not candidates:
            raise ValueError(f"No CSV/TXT table found inside {zip_path.name}")
        member = candidates[0]
        with zf.open(member) as f:
            return pd.read_csv(f, nrows=nrows, low_memory=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with simple uppercase underscore column names."""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return out


def core_field_check(df: pd.DataFrame) -> dict[str, list[str]]:
    """Report expected core fields found and missing after normalization."""
    cols = set(normalize_columns(df).columns)
    found = sorted(EXPECTED_CORE_FIELDS & cols)
    missing = sorted(EXPECTED_CORE_FIELDS - cols)
    return {"found": found, "missing": missing}
