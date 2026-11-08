# Decision-Oriented Validation of Shortened Personality Inventories

This repository accompanies the manuscript **“Decision-Oriented Validation of Shortened Personality Inventories: Rank Fidelity, Tail Agreement, and Two-Stage Screening.”** It provides the analysis scripts, non-identifying derived outputs, source data for Figure 6, and publication-quality figures used in the submission.

The study evaluates how progressive shortening of four public personality inventories affects rank-order agreement, recovery of full-form top-decile cases, fixed-cut calibration, and two-stage screening burden. All reported screening results concern agreement with the corresponding full-form decision; they do not establish external-criterion or clinical accuracy.

## Contents

- `analysis/` — the primary fixed-form holdout analysis, `v21_fixed_form_holdout_validation_20261107.py`, and `psych_gate_r5e_tiesafe_curves.py`, which supplies the core scoring utilities.
- `results/` — frozen, non-identifying JSON outputs supporting reported results.
- `figure_source_data/` — source data for the two-stage screening figure.
- `figures/` — six 600 dpi PNG figures prepared for submission.
- `data/` — instructions for obtaining the public raw data; raw respondent files are deliberately not included.

## Reproduce the fixed-form holdout analysis

1. Create a Python environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Download and unpack the four public source archives as described in [`data/README.md`](data/README.md), or set `SHORTFORM_DATA_DIR` to their parent directory.

3. Run:

   ```bash
   python analysis/v21_fixed_form_holdout_validation_20261107.py
   ```

The script writes `results/v21_fixed_form_holdout_validation_20261107.json`. Fixed random seeds are embedded in the script. Full reproduction may be computationally demanding because the procedure includes 200 fixed random-form comparisons and 1,000 bootstrap resamples.

## Data source and reuse

The anonymous raw data and codebooks are available from the [Open-Source Psychometrics Project raw-data archive](https://openpsychometrics.org/_rawdata/). No raw respondent data are redistributed here. Users should follow the source platform’s terms and preserve its codebooks when working with the data.

## Scope and status

This repository is an anonymous, review-stage reproducibility resource. It intentionally contains no author identity, manuscript title page, participant-level data, or confidential files. The archived JSON output preserves the fixed-form holdout results reported in the manuscript.
