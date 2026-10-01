# Getting started

You will need a terminal, an internet connection for the first package installation, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).
Download or clone the repository, then open a terminal in its top-level folder. The folder should
contain `pyproject.toml`, `uv.lock` and `README.md`.

## Install the environment

```bash
uv sync --locked
uv run traffic-study check
```

The lockfile supplies exact package versions for the selected Python and platform. The default
Python is 3.14. The environment lives in `.venv`, which is excluded from Git.
The status command reports whether the dataset and generated outputs are available. Outputs marked
“not run yet” are expected on a fresh checkout.

## Run one stage at a time

Run each command after the preceding one finishes.

| Command | What it does | Output to inspect |
| --- | --- | --- |
| `uv run traffic-study prepare` | Audits records and creates chronological splits | `reports/results/data_audit.json` |
| `uv run traffic-study train` | Fits both models and calibrates thresholds | `reports/results/calibration.csv` |
| `uv run traffic-study validate` | Runs a validation pilot and records inputs | `artifacts/frozen_protocol.json` |
| `uv run traffic-study evaluate` | Runs paired test scenarios and sensitivity checks | `reports/results/summary.csv` |
| `uv run traffic-study report` | Builds charts and a written interpretation | `reports/RESULTS.md` |
| `uv run pytest -q` | Tests data handling, models, injections and metrics | Terminal test summary |

Preparation should report 48,204 source rows and 40,575 unique observed hours. Evaluation with the
default settings produces 85 paired scenarios and 680 metric rows: 85 × two methods × four thresholds.
These are repeated measurements of 85 datasets, not 680 independent datasets.

`uv run traffic-study all` runs the five study stages in order. Reruns replace generated outputs in
place. The original download and the published reference results remain available.

## Use the notebooks

```bash
uv run jupyter lab
```

Keep the terminal running while using JupyterLab. Open `notebooks/01_understand_the_data.ipynb` and
select **Python 3 (ipykernel)**. Click a code cell and press **Shift+Enter** to run it and select the
next cell. Read the explanations between cells.

1. Notebook 01 inspects the source, gaps and split dates.
2. Notebook 02 walks through both detectors and their score thresholds.
3. Notebook 03 changes a copy of the validation data and measures detection.
4. Notebook 04 reads the final experiment outputs.

Run the study before starting the notebooks. To repeat a notebook from a clean session, use
**Kernel → Restart Kernel and Run All Cells**. Stop JupyterLab with **Ctrl+C** in its terminal.

For an automated notebook check:

```bash
uv run python scripts/execute_notebooks.py
uv run python scripts/verify_results.py
```

Executed copies are written to `reports/executed_notebooks/`; the source notebooks stay clean.
The verifier checks saved hashes, reconstructs the injected records and checks the executed copies.

## Common problems

- **`uv` is not recognised:** complete the uv installation and open a new terminal.
- **A generated file is missing:** run the earlier stages, or use `uv run traffic-study all`.
- **Frozen inputs changed:** rerun training, validation and evaluation in order. Record the design change.
- **A notebook cannot import the project:** launch JupyterLab through `uv run jupyter lab` in the repository.
- **The browser did not open:** use the local URL printed by JupyterLab in the terminal.
- **No network connection:** installed dependencies and the included raw dataset allow an offline run.

The optional `setup.sh` and `run.sh` wrappers accept the same stages in a Bash shell. The `uv` commands
above are the main interface and also work in PowerShell.
