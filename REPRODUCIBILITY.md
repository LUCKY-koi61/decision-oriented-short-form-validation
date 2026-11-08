# Reproducibility notes

## Software

The analyses were run with Python and the packages listed in `requirements.txt`. The repository does not pin a platform-specific interpreter build; record the Python version used for any independent rerun.

## Data layout

The runnable scripts expect the four provider archives to be unpacked under `data/` as documented in `data/README.md`. To use an existing copy elsewhere, export `SHORTFORM_DATA_DIR` before running a script:

```bash
export SHORTFORM_DATA_DIR=/absolute/path/to/data
python code/experiments/v21_fixed_form_holdout_validation_20261107.py
```

## Checks

- The JSON output under `outputs/validation/` is a frozen, non-identifying study output.
- `code/experiments/v21_fixed_form_holdout_validation_20261107.py` is the primary runnable analysis for the fixed-form, respondent-holdout results reported in Table 9 and Supplementary Table S2.
- The accompanying manuscript and supplementary material report the remaining analysis results; the present public repository focuses on the fixed-form holdout analysis and its direct output.

## Boundaries

The repository supports computational reproduction of the reported full-form-referenced analyses. It does not distribute participant-level data, external clinical outcomes, or a clinical screening instrument.
