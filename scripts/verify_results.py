"""Verify saved experimental outputs independently of the execution loop."""

import hashlib
import json
from datetime import datetime, timezone

import nbformat
import numpy as np
import pandas as pd

from traffic_anomaly.data import load_split
from traffic_anomaly.experiments import scenarios as scenario_specs
from traffic_anomaly.experiments import verify_frozen
from traffic_anomaly.models import load_models
from traffic_anomaly.paths import RESULTS, ROOT, config, sha256, write_json

verify_frozen()
models, thresholds = load_models()
manifest = json.loads((RESULTS / "run_manifest.json").read_text())
for name, expected in manifest["result_sha256"].items():
    assert sha256(RESULTS / name) == expected, f"Changed result: {name}"
test = load_split("test")
all_metrics = pd.read_csv(RESULTS / "metrics.csv")
scenarios = pd.read_csv(RESULTS / "scenario_manifest.csv")
changes = pd.read_csv(RESULTS / "injected_changes.csv.gz")
expected_count = len(list(scenario_specs(config())))
assert len(scenarios) == manifest["n_test_scenarios"] == expected_count
rows_per_scenario = len(models) * len(config()["alert_rates"])
assert len(all_metrics) == manifest["n_metric_rows"] == expected_count * rows_per_scenario
assert (all_metrics.groupby("scenario").size() == rows_per_scenario).all()
assert (all_metrics.groupby("scenario").fingerprint.nunique() == 1).all()
for _, scenario in scenarios.iterrows():
    log = changes[changes.scenario == scenario.scenario]
    positions = log.position.to_numpy()
    assert len(log) == scenario.n_changed
    assert not log.position.duplicated().any()
    np.testing.assert_array_equal(test.traffic_volume.iloc[positions], log.original_value)
    values = test.traffic_volume.to_numpy(dtype=np.int64).copy()
    values[positions] = log.altered_value.to_numpy(dtype=np.int64)
    labels = (values != test.traffic_volume.to_numpy()).astype(np.int8)
    digest = hashlib.sha256()
    digest.update(test.date_time.astype("int64").to_numpy().tobytes())
    digest.update(values.tobytes())
    digest.update(labels.tobytes())
    assert digest.hexdigest() == scenario.fingerprint
    matching = all_metrics[all_metrics.scenario == scenario.scenario]
    assert (matching.n_injected == labels.sum()).all()
    assert (matching.true_positive + matching.false_negative == matching.n_injected).all()
    assert (
        matching.benchmark_false_positive + matching.true_negative == matching.n_background
    ).all()
for family in config()["families"]:
    example = pd.read_csv(RESULTS / f"example_{family}.csv", parse_dates=["date_time"])
    for name, model in models.items():
        scores = model.score(example)
        np.testing.assert_allclose(scores, example[f"{name}_score"], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(
            scores > thresholds[name][str(config()["primary_alert_rate"])], example[f"{name}_alert"]
        )
notebooks = []
executed = ROOT / "reports/executed_notebooks"
expected_names = {p.name for p in (ROOT / "notebooks").glob("*.ipynb")}
assert {p.name for p in executed.glob("*.ipynb")} == expected_names, "Execute notebooks first."
for path in sorted(executed.glob("*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    code = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    assert not any(out.output_type == "error" for cell in code for out in cell.outputs)
    notebooks.append({"file": path.name, "executed_code_cells": len(code)})
proof = {
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "checks_passed": True,
    "reconstructed_scenarios": len(scenarios),
    "paired_metric_rows": len(all_metrics),
    "rescored_saved_examples": 4,
    "verified_notebooks": notebooks,
    "uv_lock_sha256": sha256(ROOT / "uv.lock"),
    "run_manifest_sha256": sha256(RESULTS / "run_manifest.json"),
}
write_json(RESULTS / "verification.json", proof)
print(json.dumps(proof, indent=2))
