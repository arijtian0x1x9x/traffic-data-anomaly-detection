"""Central paths so scripts and notebooks find the same project resources."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("TRAFFIC_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"
ARTIFACTS = ROOT / "artifacts"


def prepare_dirs():
    for path in [RAW, PROCESSED, RESULTS, FIGURES, ARTIFACTS]:
        path.mkdir(parents=True, exist_ok=True)


def config():
    return json.loads((ROOT / "configs" / "study.json").read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")
