# Reproducing and extending the results

The reference run uses the supplied UCI data, `configs/study.json`, Python 3.14 and the locked
dependencies. The public packaging changes the setup and documentation, not the model definitions
or injection recipes. Numerical results are checked against the included reference summary.

## Repeat the experiment

```bash
uv sync --locked
uv run traffic-study all
uv run pytest -q
uv run python scripts/execute_notebooks.py
uv run python scripts/verify_results.py
```

Verification checks the frozen inputs and result hashes, reconstructs all altered sequences from
their logs, rescores saved examples and confirms that each notebook completed. Its record is written
to `reports/results/verification.json`.

A repeat run with the same versions and seeds should reproduce the numerical summary. Timings and
run timestamps will differ. Different Python versions may resolve different package versions from
the lockfile, so compare their results rather than assuming bit-for-bit equality.

## What is recorded

- The UCI archive and compressed CSV hashes in `data/raw/source.json`.
- Split sizes, dates and data rules in `reports/results/data_audit.json`.
- Trained-model and threshold hashes in `artifacts/training_manifest.json`.
- Settings, split hashes and relative source-file hashes in `artifacts/frozen_protocol.json`.
- Package versions, counts and result hashes in `reports/results/run_manifest.json`.

Run records omit usernames, home directories, hostnames and operating-system fingerprints. The
published notebooks contain input cells only. Automated execution saves output to an ignored folder.

## Try a follow-up

Start by changing the injection seed or severity in notebook 03. It works on a copy of validation
data, which keeps the saved study outputs separate from the exercise.

For a full study change, edit `configs/study.json`, explain the reason and rerun the stages. If split
fractions or cleaning rules change, start with preparation. If models or thresholds change, start
with training. Validation records the new inputs before evaluation can use them.

The existing test period has already been examined. A method chosen after viewing those results is
an exploratory follow-up. Use another period or dataset for a fresh final assessment.

## Before sharing changes

```bash
uv run ruff check src scripts tests
uv run python scripts/clear_notebook_outputs.py
uv run python scripts/check_notebooks.py
uv run python scripts/check_public_files.py
```

The public-file check is a practical check for common accidental disclosures, not a guarantee that
every possible sensitive value can be recognised. Review the actual files being committed too.
