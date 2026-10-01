# Project structure

Run commands from the repository root: the folder containing `pyproject.toml`.

```text
traffic-data-anomaly-detection/
├── README.md                 Research question, setup and headline results
├── CONTRIBUTING.md           How to propose fixes and new experiments
├── LICENSE                   Code and documentation licence
├── pyproject.toml            Package metadata and dependencies
├── uv.lock                   Resolved dependency versions
├── configs/study.json         Experiment settings
├── data/
│   ├── README.md             Data attribution and processing notes
│   ├── raw/                  Original UCI files and provenance
│   └── processed/            Generated observations and chronological splits
├── src/traffic_anomaly/
│   ├── paths.py              Repository paths and configuration
│   ├── data.py               Download, audit and split
│   ├── features.py           Inputs supplied to Isolation Forest
│   ├── models.py             Both detectors and threshold calibration
│   ├── injection.py          Controlled changes and their labels
│   ├── evaluation.py         Per-hour and event metrics
│   ├── experiments.py        Validation protocol and test scenarios
│   ├── reporting.py          Charts and written results
│   └── cli.py                Stage commands
├── notebooks/                Four worked notebooks, in reading order
├── scripts/                  Stage runners and verification helpers
├── tests/                    Automated correctness tests
├── artifacts/                Generated fitted models and protocol records
├── reports/
│   ├── reference/            Published numerical examples and six charts
│   ├── results/              Generated metrics, audits and change logs
│   ├── figures/              Generated charts
│   └── executed_notebooks/   Generated notebook execution copies
├── docs/                     Setup, methods and reproducibility guides
└── .github/workflows/        Automated tests for pull requests and pushes
```

Generated directories appear when their stages run. Git ignores them, along with environments,
caches and editor files. The reference results are kept separately so a rerun does not overwrite them.

## Follow one result back to the data

1. Find a scenario and method in `reports/results/metrics.csv`.
2. Check its fingerprint in `scenario_manifest.csv` to identify the altered sequence.
3. Read its original and altered counts in `injected_changes.csv.gz`.
4. Find event boundaries in `injected_events.csv`.
5. Check the input splits and hashes recorded in `artifacts/frozen_protocol.json`.

`summary.csv` aggregates the main results across injection seeds. `original_alert_rates.csv` records
alerts on unchanged test data. `model_seed_sensitivity.csv` varies forest seeds with fixed injections.
`subgroups.csv` keeps counts alongside metrics for each month, hour and weekday/weekend group.

The numbered scripts mirror the five CLI stages. `execute_notebooks.py` writes executed copies;
`verify_results.py` reconstructs scenarios and checks their saved evidence. `check_notebooks.py`
checks that source notebooks are valid and contain no saved output. `check_public_files.py` checks
shareable files for common local-path and credential patterns.
