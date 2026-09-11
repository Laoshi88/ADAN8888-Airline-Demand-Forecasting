# ADAN8888 Applied Analytics Project — Airline Passenger Demand Forecasting

## Week 1 objective
Forecast next-month passenger demand for a directional U.S. domestic scheduled-passenger carrier-route using BTS T-100 Domestic Segment (All Carriers).

**Unit of analysis:** `UniqueCarrier × Origin × Dest × Year-Month` after filtering to Service Class `F` and aggregating aircraft-level records.

**Target:** passengers in month `t+1`.

## Repository structure
- `data/raw/` — immutable BTS source extract and source notes
- `data/processed/` — later cleaned/model-ready datasets
- `documents/week_01/` — Week 1 report
- `notebooks/` — weekly Jupyter notebooks beginning Week 2
- `models/` — serialized candidate/final models
- `src/` — reusable Python code
- `outputs/figures/` — report figures

## Week 1 status
- [x] Problem statement and modeling intent defined
- [x] Business value articulated
- [x] Assumption-based economic value calculated
- [x] 13-week plan aligned to the course syllabus
- [x] Dataset and supervised-regression framing documented
- [ ] Download the selected BTS extract into `data/raw/` from the official TranStats interface
- [ ] Confirm JupyterHub access
- [ ] Keep repository **private** and add the instructor as collaborator

## Data source
U.S. DOT Bureau of Transportation Statistics, TranStats: **T-100 Domestic Segment (All Carriers)**.
