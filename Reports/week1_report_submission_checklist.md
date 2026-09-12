# Week 1 Submission Checklist

## Written report (100 points)
- [x] Problem statement
- [x] Articulation of value
- [x] Potential economic value calculation
- [x] Explicit documented assumptions / footnotes
- [x] Calculation shown step by step
- [x] 13-week project plan aligned to the syllabus
- [x] Dataset description and source
- [x] Explanation of how the dataset addresses the problem
- [x] Modeling type identified as supervised regression
- [x] Professional table(s) and economic-value sensitivity figure
- [x] Report remains within the 10-page maximum
- [x] 11-point professional font and 1.5-spaced body paragraphs
- [x] Final PDF stored under `reports/`

## Repository standards from the course Git/GitHub instructions
- [x] Use `data/` for datasets (with `raw/` and `processed/` subfolders)
- [x] Use `models/` for serialized model objects later in the semester
- [x] Use `notebooks/` for Jupyter notebooks
- [x] Use `reports/` for written reports and supporting report artifacts
- [x] Use `src/` for Python source code
- [x] Keep `README.md` at the root for project description/navigation
- [x] Keep `requirements.txt` at the root for Python dependencies
- [x] Do not use the Cookiecutter `docs/` or `references/` folders for this course
- [x] Apply the `week#_type_description` naming convention to weekly files
- [ ] Place the official BTS T-100 raw extract(s) in `data/raw/`
- [ ] Confirm repository visibility is **PRIVATE**
- [ ] Add instructor collaborator: GitHub `savasnurtekin` (or course email `savasnu@bc.edu`)
- [ ] Confirm JupyterHub / Python environment access
- [ ] Push the final Week 1 structure to the private GitHub repository
- [ ] Submit the private GitHub repository URL in Canvas

## Data-size reminder from the course instructions
Keep the dataset manageable for the course workflow. The course Git/GitHub handout specifically recommends keeping the dataset below 2 GB where possible; if a dataset cannot be pushed directly, retain it locally and document/source it by URL as instructed.

## Modeling guardrail
Do not use realized month `t+1` seats/departures as predictors of month `t+1` passengers unless they were genuinely available at the forecast origin. Default to lagged capacity/frequency variables to prevent target leakage.
