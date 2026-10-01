"""Small commands match the numbered learning stages."""

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version

from .paths import RAW, RESULTS, ROOT


def check():
    print(f"Project configuration found: {(ROOT / 'configs/study.json').is_file()}")
    print(f"Python: {sys.version.split()[0]}")
    for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "jupyterlab"]:
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "not installed (optional for command-line use)"
        print(f"  {name}: {installed}")
    print(f"Raw data present: {(RAW / 'Metro_Interstate_Traffic_Volume.csv.gz').exists()}")
    for name in ["data_audit.json", "calibration.csv", "validation_pilot.csv", "run_manifest.json"]:
        print(f"  {name}: {'ready' if (RESULTS / name).exists() else 'not run yet'}")
    if (RESULTS / "run_manifest.json").exists():
        m = json.loads((RESULTS / "run_manifest.json").read_text())
        print(f"Last completed evaluation: {m['n_test_scenarios']} paired scenarios.")


def main():
    parser = argparse.ArgumentParser(description="Traffic anomaly study: run one stage at a time.")
    parser.add_argument(
        "stage", choices=["check", "prepare", "train", "validate", "evaluate", "report", "all"]
    )
    args = parser.parse_args()
    from .data import prepare
    from .experiments import evaluate, validate
    from .models import train
    from .reporting import report

    actions = {
        "check": check,
        "prepare": prepare,
        "train": train,
        "validate": validate,
        "evaluate": evaluate,
        "report": report,
    }
    if args.stage == "all":
        for stage in ["prepare", "train", "validate", "evaluate", "report"]:
            print(f"\n--- {stage.upper()} ---", flush=True)
            actions[stage]()
    else:
        actions[args.stage]()


if __name__ == "__main__":
    main()
