# PureTarget designer

Streamlit app for designing PacBio PureTarget tiled CRISPR capture panels on **hg38**.
Not an official PacBio tool.
This app is a thin UI wrapper around the existing stepwise pipeline scripts in `scripts/`. It is designed to run from the repository root and execute the current UCSC/Ensembl-backed workflow end to end.

## What the app does

Inputs:
- locus coordinates in `chr:start-end` format
- target tile size
- min tile size
- max tile size
- single-tile toggle

Outputs:
- downloadable `guides_even.fasta`
- downloadable `guides_odd.fasta`
- guide order table preview
- iterative redesign round summary
- candidate tile geometry plot

## Repository layout expected by the app

The app assumes:

- the app entrypoint is `streamlit_app.py`
- the pipeline scripts live in `scripts/`
- output runs are written under `streamlit_runs/`

Minimal layout:

```text
your_repository/
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── README.md
├── streamlit_app.py
└── scripts/
    ├── 01_parse_locus.py
    ├── 02_make_tiles.py
    ├── ...
    └── 11c_generate_human_readable_report.py
```

## Local run

Create or activate your environment, install the Python dependencies, and launch Streamlit:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL shown in the terminal.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set the **main file path** to:

```text
streamlit_app.py
```

4. Pick the Python version that matches your development environment as closely as possible.
5. Deploy.

## Important deployment note

Streamlit Community Cloud supports `requirements.txt` for Python dependencies and `.streamlit/config.toml` for app configuration. If a repository contains multiple dependency files, Community Cloud only uses the first supported dependency file it finds, so keep the repo simple and prefer a single `requirements.txt`.

If this repository still has an old `environment.yml` or `environment.yaml` that you no longer want to use for deployment, remove it or rename it so there is no ambiguity. Community Cloud documents that only one dependency file is used.

## Required Python packages

This app and the current simplified pipeline need:

- `streamlit`
- `pandas`
- `matplotlib`
- `requests`

If your patched pipeline scripts import any additional packages, add them to `requirements.txt` before deploying.

## Configuration

The included `.streamlit/config.toml` file sets a basic wide layout and hides some development noise. It is optional but supported when placed at `.streamlit/config.toml` in the repository root.

## Notes for this app

- The app runs the pipeline scripts with `subprocess` and writes each run to a temporary subdirectory inside `streamlit_runs/`.
- No secrets file is required for the current UCSC/Ensembl-backed workflow.
- The app assumes it is launched from the repository root.

## Troubleshooting

### App deploys but fails immediately
Check:
- the app entrypoint path is `streamlit_app.py`
- the `scripts/` directory is present in the repo
- `requirements.txt` contains all imports used by both the app and the pipeline

### Dependency issues on Streamlit Cloud
Keep only one dependency file and use `requirements.txt` at the repo root unless you have a specific reason not to. Community Cloud explicitly supports this layout.

### The app runs locally but not in Cloud
Make sure the Python version selected during deployment is compatible with the versions you used locally. Streamlit recommends matching the deployment Python version to development when dependencies are version-sensitive.

## Files included for deployment

- `README.md`
- `requirements.txt`
- `.streamlit/config.toml`
