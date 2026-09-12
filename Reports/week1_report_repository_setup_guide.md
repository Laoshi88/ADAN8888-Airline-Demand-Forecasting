# Week 1 Repository Setup Guide

This guide implements the course's **Getting Started with Git and GitHub** standards for the ADAN8888 airline-demand project.

## Recommended repository
- GitHub profile: `Laoshi88`
- Recommended private repository name: `ADAN8888-Airline-Demand-Forecasting`
- Recommended remote URL after creation: `https://github.com/Laoshi88/ADAN8888-Airline-Demand-Forecasting.git`

Your local folder may remain named `adan8888_week1_project`; the local folder name does not need to match the GitHub repository name.

## Course folder structure
Use these folders from the course/Cookiecutter template:
- `data/` — datasets. This project uses `data/raw/` and `data/processed/`.
- `models/` — serialized model objects later in the course.
- `notebooks/` — Jupyter notebooks.
- `reports/` — weekly written reports and report-supporting artifacts.
- `src/` — Python source code.
- `README.md` — project description and navigation.
- `requirements.txt` — libraries/dependencies needed to run the project.

Do **not** use the Cookiecutter `docs/` or `references/` folders for this course.

## File naming convention
Use `week#_type_description` consistently. Examples in this project:
- `week1_data_t100_domestic_segment_all_carriers_2015.zip`
- `week1_notebook_environment_check.ipynb`
- `week1_report_identify_problem_statement_and_dataset.pdf`
- `week1_src_download_t100_domestic_segment.py`

## First GitHub push from VS Code / PowerShell
Run from the project root:

```powershell
git init
git branch -M main
git remote add origin https://github.com/Laoshi88/ADAN8888-Airline-Demand-Forecasting.git
git add -A
git status
git commit -m "Week 1: define airline demand forecasting project and dataset"
git push -u origin main
```

If `git remote -v` already shows an old/team repository, remove it first with:

```powershell
git remote remove origin
```

Then add the `Laoshi88` remote shown above.

## Instructor collaborator
On GitHub: **Repository → Settings → Collaborators → Add people**. Search for `savasnurtekin` (or `savasnu@bc.edu`) and send the invitation. Keep the repository **private**.

## Raw data
Run the downloader from the project root:

```powershell
python .\src\week1_src_download_t100_domestic_segment.py --start-year 2015 --end-year 2026 --out .\data\raw
```

The downloader preserves annual source ZIPs unchanged in `data/raw/` and writes a manifest. Do not clean or alter raw files in place.
