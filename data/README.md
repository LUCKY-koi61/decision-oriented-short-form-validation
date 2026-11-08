# Data directory

Raw respondent data are not redistributed in this repository. Download the anonymous source archives from the [Open-Source Psychometrics Project raw-data archive](https://openpsychometrics.org/_rawdata/) and unpack them so that this directory contains:

```text
data/
├── HEXACO/HEXACO/data.csv
├── BIG5/BIG5/data.csv
├── NPI/NPI/data.csv
└── SD3/SD3/data.csv
```

Alternatively, set `SHORTFORM_DATA_DIR` to an existing directory with that layout. Do not commit raw respondent data or codebooks to this repository.
