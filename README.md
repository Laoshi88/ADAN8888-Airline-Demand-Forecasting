# ADAN8888 Applied Analytics Project — Airline Passenger Demand Forecasting

## Week 1 Objective

Develop the foundation for a supervised regression project that forecasts next-month passenger demand for a directional U.S. domestic scheduled-passenger carrier-route using BTS **T-100 Domestic Segment (All Carriers)** data.

**Unit of analysis:** `UniqueCarrier × Origin × Dest × Year-Month` after filtering to service class `F` and aggregating aircraft-level records.

**Target:** Passengers in month `t+1`.

## GitHub Repository

- GitHub profile: `Laoshi88`
- Repository name: `ADAN8888-Airline-Demand-Forecasting`
- Instructor collaborator: `savasnurtekin` (or `savasnu@bc.edu`)

## Course-Standard Repository Structure

The structure below follows the course's Git/GitHub instructions. The Cookiecutter `docs/` and `references/` folders are intentionally not used.

- `Data/` — all project datasets
  - `Data/raw/` — immutable BTS source downloads and source/download documentation
  - `Data/processed/` — cleaned, filtered, aggregated, and later model-ready datasets
- `Models/` — serialized model objects later in the semester
- `Notebooks/` — Jupyter notebooks
- `Reports/` — weekly written reports and supporting report artifacts
- `Src/` — Python source code
- `README.md` — project description and repository navigation
- `requirements.txt` — Python dependencies required to run the project

## File Naming Convention

Weekly artifacts follow the instructor's `week#_type_description` convention. Examples:

- `week1_data_t100_domestic_segment_all_carriers_2015.zip`
- `week1_notebook_environment_check.ipynb`
- `week1_report_identify_problem_statement_and_dataset.pdf`
- `week1_src_download_t100_domestic_segment.py`

## Week 1 Artifacts Included

- `Reports/week1_report_identify_problem_statement_and_dataset.pdf` — submission-ready Week 1 written report
- `Reports/week1_report_identify_problem_statement_and_dataset.docx` — editable report source
- `Reports/week1_report_economic_value_assumptions.csv` — scenario assumptions used in the report
- `Reports/week1_report_economic_value_results.csv` — reproducible economic-value analysis results
- `Reports/week1_report_t100_selected_fields.md` — planned T-100 field reference
- `Reports/week1_report_submission_checklist.md` — final Week 1 checklist
- `Reports/week1_report_repository_setup_guide.md` — GitHub/course-structure setup guide
- `Reports/week1_report_figures/week1_report_economic_value_sensitivity.png` — economic-value sensitivity visualization
- `Notebooks/week1_notebook_environment_check.ipynb` — environment and repository validation notebook
- `Src/week1_src_download_t100_domestic_segment.py` — reproducible BTS annual-data downloader
- `Src/week1_src_economic_value_analysis.py` — reproducible Week 1 economic-value analysis
- `Data/raw/week1_data_t100_download_instructions.md` — source and manual-download fallback instructions
- `Data/raw/week1_data_t100_download_manifest.csv` — manifest of downloaded BTS T-100 source files

## Week 1 Status

- [x] Problem statement and modeling intent defined
- [x] Business value articulated
- [x] Assumption-based economic value calculated
- [x] 13-week project plan aligned to the course syllabus
- [x] Dataset and supervised-regression framing documented
- [x] Repository restructured to course standards
- [x] Weekly files renamed using the course naming convention
- [x] Official BTS extracts downloaded into `Data/raw/`
- [x] JupyterHub/Python environment access confirmed
- [x] Private GitHub repository created under `Laoshi88`
- [x] Instructor collaborator added (`savasnurtekin`)
- [x] Final Week 1 artifacts pushed to GitHub
- [x] Repository link prepared for submission in Canvas

## Data Download

The raw dataset consists of annual extracts from the BTS **T-100 Domestic Segment (All Carriers)** table for 2015 through 2026.

To recreate the raw downloads from the project root in VS Code or PowerShell, run:

```powershell
python .\Src\week1_src_download_t100_domestic_segment.py --start-year 2015 --end-year 2026 --out .\Data\raw
```

The downloader stores the source files in `Data/raw/` and creates a download manifest for reproducibility.

Raw ZIP files should remain unchanged. Any cleaning, filtering, aggregation, or feature engineering should be performed on copies and stored under `Data/processed/`.

## Week 1 Economic-Value Analysis

The Week 1 business case uses documented assumptions to estimate the potential economic value of improved passenger-demand forecasting.

To reproduce the Week 1 calculations and sensitivity visualization, run:

```powershell
python .\Src\week1_src_economic_value_analysis.py --assumptions .\Reports\week1_report_economic_value_assumptions.csv --results .\Reports\week1_report_economic_value_results.csv --figure .\Reports\week1_report_figures\week1_report_economic_value_sensitivity.png
```

The script regenerates the economic-value results and confirms consistency with the assumptions documented in the written report.

## Week 1 Notebook Note

The Week 1 assignment does not require a submitted modeling or empirical-analysis notebook/code output. Notebook and code-output deliverables begin in Week 2.

The included `Notebooks/week1_notebook_environment_check.ipynb` is therefore used as a professional environment and repository validation check rather than as a Week 1 modeling analysis.

Empirical ingestion, exploration, preprocessing, and modeling of the BTS T-100 observations will begin in subsequent course weeks in accordance with the course schedule.