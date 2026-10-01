# Contributing

Fixes, clearer explanations and well-described experiments are welcome. For a bug report, include
the command, package versions and the smallest example that reproduces the problem. Remove local
paths and credentials from logs before sharing them.

## Make a change

1. Fork or clone the repository and create a branch for the change.
2. Run `uv sync --locked` to install the environment.
3. Keep a code change focused and explain why it is needed.
4. Add a test when the change affects model behaviour, data handling or evaluation.
5. Run the checks below, then open a pull request with a description and test results.

```bash
uv run pytest -q
uv run ruff check src scripts tests
uv run python scripts/clear_notebook_outputs.py
uv run python scripts/check_notebooks.py
uv run python scripts/check_public_files.py
```

For an experimental change, also rerun the study and describe the data, settings and limitations.
Keep comparisons paired: both methods must receive the same altered records. Do not choose settings
using final test performance and then describe that test as untouched.

Keep documentation concrete. Explain what a command does, which file it produces and how a reader
can check the result. Put large generated outputs in an appropriate release or external data store
rather than adding environments, caches or fitted model binaries to the repository.

GitHub Actions is configured to run checks on Linux and Windows with Python 3.11 and 3.14. Its status
becomes available after the repository is published and the workflow runs.
