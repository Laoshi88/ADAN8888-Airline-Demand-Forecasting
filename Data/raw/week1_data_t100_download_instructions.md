# Week 1 Data Download Instructions — BTS T-100 Domestic Segment (All Carriers)

## Source selected for the semester project
U.S. Department of Transportation, Bureau of Transportation Statistics (BTS), TranStats: **T-100 Domestic Segment (All Carriers)**.

## Recommended initial scope
- January 2015 through the latest uniform month available at download time.
- During processing, filter the core analytical sample to Service Class `F` (scheduled passenger/cargo service).
- Preserve the raw downloads unchanged in this folder.
- Save all cleaned, filtered, aggregated, or model-ready data under `data/processed/`.

## Preferred automated download
From the repository root:

```powershell
python .\src\week1_src_download_t100_domestic_segment.py --start-year 2015 --end-year 2026 --out .\data\raw
```

The script writes annual ZIP files using the course naming convention plus `week1_data_t100_download_manifest.csv`.

## Core fields to retain/validate
`Year`, `Quarter`, `Month`, `UniqueCarrier`, `AirlineID`, `UniqueCarrierName`, `OriginAirportID`, `Origin`, `OriginCityName`, `OriginState`, `DestAirportID`, `Dest`, `DestCityName`, `DestState`, `AircraftType`, `AircraftConfig`, `Class`, `Passengers`, `Seats`, `DepScheduled`, `DepPerformed`, `Distance`, and `LoadFactor` (or recompute it).

## Manual fallback
If the BTS form changes and the script cannot retrieve the files, use the official TranStats T-100 Domestic Segment (All Carriers) interface. Select Geography = All, one year at a time, Period = All, select all fields, and download ZIP output. Rename downloaded files to the same `week1_data_...` convention before placing them here.

Source landing page: <https://www.transtats.bts.gov/Tables.asp?QO_VQ=EEE>

## Course repository-size note
The course Git/GitHub instructions recommend a dataset smaller than 2 GB where possible. If the data cannot be stored directly in the repository, keep the source data locally and document the official URL/source in the repository, following the instructor's directions.
