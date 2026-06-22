# PureTarget designer

Streamlit app for designing PacBio PureTarget tiled CRISPR capture panels on **hg38**.
Not an official PacBio tool.
This app is a thin UI wrapper around the existing stepwise pipeline scripts in `scripts/`. It is designed to run from the repository root and execute the current UCSC/Ensembl-backed workflow end to end.

## What the app does

The app has two design modes.

### Multi-tile locus mode

Use this mode when you want to tile a continuous locus.

Inputs:
- locus coordinates in `chr:start-end` format
- target tile size
- min tile size
- max tile size

In this mode, the app keeps the original tiled-locus behavior: it designs a set of overlapping tiles across the locus, fetches candidate guide sequence from UCSC, scores and pairs guides, detects problematic tiles, and tries iterative redesigns when needed.

### Hotspot / variant list mode

Use this mode when you want to capture specific variants or variant-containing regions that may be separated by large gaps.

Inputs:
- target regions, one per line
- coordinate system: `0-based half-open` or `1-based inclusive`
- target tile size
- max tile size
- minimum tile gap

Accepted target formats:

```text
chr9:108931142-108931143
chr9 108931142 108931143 ELP1:c.4C>T
chr9	108931142	108931143	ELP1:c.4C>T
```

Hotspot mode currently requires all target regions to be on one chromosome. If you want to capture an entire gene in hotspot mode, enter the gene interval as the target region.

The hotspot planner tries to cover all submitted target regions with the smallest number of non-overlapping tiles. Each tile must be no larger than `Max tile size`; neighboring tiles must be at least the requested minimum gap apart, which defaults to 500 bp. The full tile set is evaluated together, because a better redesign for one tile can change which targets another tile should cover.

Hotspot guide selection also enforces a target-protection rule: each guide plus PAM footprint must be at least 50 bp away from any submitted target region. Guides that violate this rule are marked unacceptable and are not used for hotspot guide pairs.

Outputs:
- multi-tile mode: downloadable `guides_even.fasta` and `guides_odd.fasta`
- hotspot mode: downloadable `guides.fasta`
- guide order table preview; hotspot mode omits odd/even pool labels
- iterative redesign round summary
- candidate tile geometry plot

## Algorithm overview

The pipeline is implemented as stepwise scripts in `scripts/` and orchestrated by `streamlit_app.py`.

At a high level, the app:

1. Parses the requested locus or hotspot targets.
2. Builds an initial tile plan.
3. Fetches reference sequence and annotation context from UCSC.
4. Finds candidate guides around tile boundaries.
5. Annotates guide quality, common SNP overlap, sequence warnings, and off-target risk.
6. Ranks guides and forms left/right guide pairs.
7. Detects problematic tiles or tile plans.
8. Runs candidate redesigns and picks the best complete design.
9. Writes FASTA files, guide order tables, plots, and a human-readable report.

UCSC fetches are cached within each app run under `streamlit_runs/<run>/ucsc_cache/`, and overlapping or adjacent UCSC windows are merged where possible to reduce repeated network requests.

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
    ├── 02b_make_hotspot_tiles.py
    ├── ...
    ├── 09h_optimize_hotspot_plans.py
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
- Hotspot target coordinates are normalized internally to 1-based inclusive coordinates before tile planning.

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
