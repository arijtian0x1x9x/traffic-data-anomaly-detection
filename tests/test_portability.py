"""Check that a checkout can use its own resources without a personal environment."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_project_root_can_be_relocated(tmp_path):
    root = tmp_path / "checkout with spaces"
    (root / "configs").mkdir(parents=True)
    (root / "configs/study.json").write_text('{"relocation_check": true}', encoding="utf-8")
    environment = os.environ.copy()
    environment["TRAFFIC_PROJECT_ROOT"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from traffic_anomaly.paths import config; print(json.dumps(config()))",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"relocation_check": True}


def test_status_output_does_not_print_local_paths(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/study.json").write_text("{}", encoding="utf-8")
    environment = os.environ.copy()
    environment["TRAFFIC_PROJECT_ROOT"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "traffic_anomaly.cli", "check"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Project configuration found: True" in result.stdout
    assert str(tmp_path) not in result.stdout
    assert sys.prefix not in result.stdout
    assert str(Path.home()) not in result.stdout
