#!/usr/bin/env python3
"""Download BTS TranStats T-100 Domestic Segment (All Carriers) annual ZIP files.

Designed for the ADAN8888 airline-demand project. It replays the public
TranStats download form, requests all fields, all U.S. geographies, and all
months for each requested year, and saves the untouched ZIP responses.

Example (run from the project root):
    python src/download_t100_domestic_segment.py --start-year 2015 --end-year 2026 --out data/raw

Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.cookiejar
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

DOWNLOAD_URL = (
    "https://www.transtats.bts.gov/DL_SelectFields.aspx"
    "?QO_fu146_anzr=Nv4%20Pn44vr45&gnoyr_VQ=GEE"
)
USER_AGENT = "Mozilla/5.0 (ADAN8888 academic project; BTS T-100 downloader)"


class TranStatsFormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden: dict[str, str] = {}
        self.field_checkboxes: list[str] = []
        self.years: set[str] = set()
        self._in_year_select = False

    def handle_starttag(self, tag: str, attrs) -> None:
        a = dict(attrs)
        tag = tag.lower()

        if tag == "input":
            input_type = a.get("type", "").lower()
            name = a.get("name")
            if input_type == "hidden" and name:
                self.hidden[name] = a.get("value", "")
            elif input_type == "checkbox" and name:
                # TranStats field checkboxes are the checkboxes that do not use
                # the site's chk* control names (Select all, ZIP, etc.).
                if not name.lower().startswith("chk"):
                    self.field_checkboxes.append(name)

        elif tag == "select" and a.get("name") == "cboYear":
            self._in_year_select = True

        elif tag == "option" and self._in_year_select:
            value = a.get("value", "")
            if re.fullmatch(r"\d{4}", value):
                self.years.add(value)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "select" and self._in_year_select:
            self._in_year_select = False


def get_form(opener, timeout: int) -> TranStatsFormParser:
    req = Request(DOWNLOAD_URL, headers={"User-Agent": USER_AGENT})
    with opener.open(req, timeout=timeout) as response:
        html = response.read().decode("utf-8", "replace")

    parser = TranStatsFormParser()
    parser.feed(html)

    if "__VIEWSTATE" not in parser.hidden:
        raise RuntimeError("Could not find __VIEWSTATE on the TranStats form.")
    if not parser.field_checkboxes:
        raise RuntimeError("Could not identify the TranStats data-field checkboxes.")
    if not parser.years:
        raise RuntimeError("Could not identify available years on the TranStats form.")
    return parser


def download_year(year: int, out_dir: Path, timeout: int = 600) -> tuple[Path, int, str]:
    jar = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))
    parser = get_form(opener, timeout)

    y = str(year)
    if y not in parser.years:
        lo, hi = min(parser.years), max(parser.years)
        raise ValueError(f"Year {year} is not offered by BTS; available form years are {lo}-{hi}.")

    form = dict(parser.hidden)
    form.update(
        {
            "btnDownload": "Download",
            "chkDownloadZip": "on",
            "cboGeography": "All",
            "cboYear": y,
            "cboPeriod": "All",
        }
    )
    for field_name in parser.field_checkboxes:
        form[field_name] = "on"

    data = urlencode(form).encode("utf-8")
    req = Request(
        DOWNLOAD_URL,
        data=data,
        headers={"User-Agent": USER_AGENT, "Referer": DOWNLOAD_URL},
    )
    with opener.open(req, timeout=timeout) as response:
        raw = response.read()

    if raw[:2] != b"PK":
        preview = raw[:500].decode("utf-8", "replace")
        raise RuntimeError(
            "BTS returned HTML/text instead of a ZIP. The download form may have changed. "
            f"Response preview: {preview!r}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"T100_Domestic_Segment_All_Carriers_{year}.zip"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    return path, len(raw), digest


def write_manifest(rows: list[dict[str, str]], out_dir: Path) -> Path:
    manifest = out_dir / "t100_download_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["year", "file", "bytes", "sha256", "downloaded_utc", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year", type=int, default=None,
                   help="Default: latest year currently offered by BTS")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds between annual requests (default 1.0)")
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    out_dir = Path(args.out)

    # A lightweight first GET determines the current latest year.
    opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
    parser = get_form(opener, args.timeout)
    latest = max(int(y) for y in parser.years)
    end_year = args.end_year if args.end_year is not None else latest

    if args.start_year > end_year:
        raise SystemExit("--start-year must be <= --end-year")

    print("BTS table: T-100 Domestic Segment (All Carriers)")
    print(f"Years requested: {args.start_year}-{end_year}")
    print(f"Latest year offered by BTS: {latest}")
    print(f"Output directory: {out_dir.resolve()}")

    rows: list[dict[str, str]] = []
    for year in range(args.start_year, end_year + 1):
        print(f"\nDownloading {year} ...", flush=True)
        path, nbytes, digest = download_year(year, out_dir, args.timeout)
        print(f"  saved {path.name} ({nbytes / (1024**2):.1f} MiB)")
        rows.append(
            {
                "year": str(year),
                "file": path.name,
                "bytes": str(nbytes),
                "sha256": digest,
                "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "source": DOWNLOAD_URL,
            }
        )
        if year != end_year:
            time.sleep(max(0.0, args.delay))

    manifest = write_manifest(rows, out_dir)
    print(f"\nDone. Manifest written to: {manifest}")
    print("Keep the ZIP files unchanged in data/raw; clean/aggregate later into data/processed.")


if __name__ == "__main__":
    main()
