# ADAN8888 Applied Analytics Project — Airline Passenger Demand Forecasting

## Week 1 objective
Develop the foundation for a supervised regression project that forecasts next-month passenger demand for a directional U.S. domestic scheduled-passenger carrier-route using BTS **T-100 Domestic Segment (All Carriers)** data.

**Unit of analysis:** `UniqueCarrier × Origin × Dest × Year-Month` after filtering to Service Class `F` and aggregating aircraft-level records.  
**Target:** passengers in month `t+1`.

## GitHub repository
- GitHub profile: `Laoshi88`
- Recommended private repository name: `ADAN8888-Airline-Demand-Forecasting`
- Repository visibility: **Private**
- Instructor collaborator to add: `savasnurtekin` (or `savasnu@bc.edu`)

## Course-standard repository structure
The structure below follows the course's Git/GitHub instructions. The Cookiecutter `docs/` and `references/` folders are intentionally not used.

- `data/` — all datasets
  - `data/raw/` — immutable BTS source downloads and source/download documentation
  - `data/processed/` — cleaned, filtered, aggregated, and later model-ready datasets
- `models/` — serialized model objects later in the semester
- `notebooks/` — Jupyter notebooks
- `reports/` — weekly written reports and supporting report artifacts
- `src/` — Python source code
- `README.md` — project description and repository navigation
- `requirements.txt` — Python dependencies required to run the project

## File naming convention
Weekly artifacts follow the instructor's `week#_type_description` convention. Examples:

- `week1_data_t100_domestic_segment_all_carriers_2015.zip`
- `week1_notebook_environment_check.ipynb`
- `week1_report_identify_problem_statement_and_dataset.pdf`
- `week1_src_download_t100_domestic_segment.py`

## Week 1 artifacts included
- `reports/week1_report_identify_problem_statement_and_dataset.pdf` — submission-ready Week 1 written report
- `reports/week1_report_identify_problem_statement_and_dataset.docx` — editable report source
- `reports/week1_report_economic_value_assumptions.csv` — scenario assumptions used in the report
- `reports/week1_report_t100_selected_fields.md` — planned T-100 field reference
- `reports/week1_report_submission_checklist.md` — final Week 1 checklist
- `reports/week1_report_repository_setup_guide.md` — GitHub/course-structure setup guide
- `notebooks/week1_notebook_environment_check.ipynb` — optional environment/repository validation notebook
- `src/week1_src_download_t100_domestic_segment.py` — reproducible BTS annual-data downloader
- `data/raw/week1_data_t100_download_instructions.md` — source and manual-download fallback instructions

## Week 1 status
- [x] Problem statement and modeling intent defined
- [x] Business value articulated
- [x] Assumption-based economic value calculated
- [x] 13-week plan aligned to the course syllabus
- [x] Dataset and supervised-regression framing documented
- [x] Repository restructured to the course standards
- [x] Weekly files renamed using the course convention
- [ ] Download the official BTS extract(s) into `data/raw/`
- [ ] Confirm JupyterHub/Python environment access
- [ ] Create/confirm the private GitHub repository under `Laoshi88`
- [ ] Add instructor collaborator (`savasnurtekin`)
- [ ] Push the final Week 1 artifacts and submit the repository link in Canvas

## Data download
From the project root in VS Code / PowerShell:

```powershell
python .\src\week1_src_download_t100_domestic_segment.py --start-year 2015 --end-year 2026 --out .\data\raw
```

Do not modify the raw ZIP files. Processed data belong under `data/processed/`.

## Week 1 notebook note
The Week 1 assignment does not require a submitted analysis notebook/code output; formal notebook/code-output deliverables begin in Week 2. The Week 1 notebook is included only as a professional environment/repository check.
