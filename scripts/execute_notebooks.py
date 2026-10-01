"""Run notebooks into a generated directory while keeping their source files clean."""

import os
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from traffic_anomaly.paths import ROOT

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "traffic-anomaly-matplotlib")
)
os.environ.setdefault(
    "JUPYTER_RUNTIME_DIR", str(Path(tempfile.gettempdir()) / "traffic-anomaly-jupyter")
)
destination = ROOT / "reports/executed_notebooks"
destination.mkdir(parents=True, exist_ok=True)
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=240,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, destination / path.name)
    print(f"Executed {path.name}", flush=True)
