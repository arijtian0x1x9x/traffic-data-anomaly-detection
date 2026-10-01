"""Freeze the protocol on validation, then evaluate identical test scenarios."""

import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import numpy as np
import pandas as pd

from .data import load_split
from .evaluation import metrics
from .injection import inject
from .models import ForestDetector, calibrate, load_models
from .paths import ARTIFACTS, PROCESSED, RESULTS, ROOT, config, sha256, write_json


def source_hashes():
    return {
        p.relative_to(ROOT).as_posix(): sha256(p)
        for p in sorted((ROOT / "src/traffic_anomaly").glob("*.py"))
    }


def run_scenario(frame, family, severity, rate, seed, cfg):
    trial = inject(frame, family, severity, rate, seed, cfg)
    if trial.labels.sum() == 0:
        raise ValueError(f"Scenario {family}/{severity} generated no actual changes.")
    return trial


def validate():
    """Pilot uses validation only. It never selects a model by its final test score."""
    cfg = config()
    models, thresholds = load_models()
    frame = load_split("validation")
    rows = []
    for family in cfg["families"]:
        trial = run_scenario(
            frame, family, "severe", cfg["primary_injection_rate"], cfg["injection_seeds"][0], cfg
        )
        for name, model in models.items():
            values = metrics(
                trial.labels,
                model.score(trial.frame),
                thresholds[name][str(cfg["primary_alert_rate"])],
                frame.date_time,
                trial.event_ids,
            )
            rows.append({"family": family, "method": name, **values})
    pd.DataFrame(rows).to_csv(RESULTS / "validation_pilot.csv", index=False)
    # Freeze the configuration, actual data partitions, model objects and source code.
    write_json(
        ARTIFACTS / "frozen_protocol.json",
        {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": cfg,
            "config_sha256": sha256(ROOT / "configs/study.json"),
            "training_manifest_sha256": sha256(ARTIFACTS / "training_manifest.json"),
            "split_sha256": {
                s: sha256(PROCESSED / f"{s}.csv") for s in ["train", "validation", "test"]
            },
            "source_sha256": source_hashes(),
            "selection": "Fixed a priori settings; no hyperparameter search or test-based selection.",
        },
    )
    print("Validation pilot complete. Protocol frozen before final test evaluation.")
    return rows


def verify_frozen():
    path = ARTIFACTS / "frozen_protocol.json"
    if not path.exists():
        raise FileNotFoundError("Run 'uv run traffic-study validate' before final evaluation.")
    frozen = json.loads(path.read_text())
    ok = frozen["config_sha256"] == sha256(ROOT / "configs/study.json")
    ok &= frozen["training_manifest_sha256"] == sha256(ARTIFACTS / "training_manifest.json")
    ok &= all(sha256(PROCESSED / f"{s}.csv") == h for s, h in frozen["split_sha256"].items())
    ok &= frozen["source_sha256"] == source_hashes()
    if not ok:
        raise ValueError(
            "Frozen inputs changed. Re-run train/validate and document the new protocol."
        )
    return frozen


def scenarios(cfg):
    for family in cfg["families"]:
        for severity in cfg["severities"]:
            for seed in cfg["injection_seeds"]:
                yield "primary", family, severity, cfg["primary_injection_rate"], seed
        for seed in cfg["injection_seeds"]:
            yield "prevalence", family, "severe", cfg["stress_injection_rate"], seed
    for seed in cfg["injection_seeds"]:
        yield "mixed", "mixed", "moderate", cfg["primary_injection_rate"], seed


def evaluate():
    frozen = verify_frozen()
    cfg = config()
    models, thresholds = load_models()
    test, val, train = [load_split(s) for s in ["test", "validation", "train"]]
    records, changes, events, manifests, subgroup_records, sensitivity = [], [], [], [], [], []
    original_scores = test.copy()
    original_rates = []
    for name, model in models.items():
        start = perf_counter()
        score = model.score(test)
        elapsed = perf_counter() - start
        original_scores[f"{name}_score"] = score
        for rate in cfg["alert_rates"]:
            threshold = thresholds[name][str(rate)]
            original_rates.append(
                {
                    "method": name,
                    "calibration_target": rate,
                    "threshold": threshold,
                    "n_hours": len(test),
                    "background_alert_rate": float(np.mean(score > threshold)),
                    "score_seconds": elapsed,
                }
            )
        original_scores[f"{name}_alert"] = score > thresholds[name][str(cfg["primary_alert_rate"])]
    original_scores.to_csv(
        RESULTS / "original_test_scores.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(original_rates).to_csv(RESULTS / "original_alert_rates.csv", index=False)
    trial_map = {}
    for suite, family, severity, rate, seed in scenarios(cfg):
        sid = f"{suite}_{family}_{severity}_{rate:g}_{seed}"
        trial = run_scenario(test, family, severity, rate, seed, cfg)
        if suite == "primary" and severity == "severe" and seed == cfg["injection_seeds"][0]:
            trial_map[family] = trial
        trial.changes.insert(0, "scenario", sid)
        trial.events.insert(0, "scenario", sid)
        changes.append(trial.changes)
        events.append(trial.events)
        manifests.append(
            {
                "scenario": sid,
                "fingerprint": trial.fingerprint,
                "eligible_hours": trial.eligible_hours,
                "n_changed": int(trial.labels.sum()),
                "changed_fraction_of_eligible": float(trial.labels.sum() / trial.eligible_hours),
            }
        )
        representative = trial.frame.copy()
        representative["original_volume"] = test.traffic_volume
        representative["injected"] = trial.labels
        representative["event_id"] = trial.event_ids
        for name, model in models.items():
            start = perf_counter()
            scores = model.score(trial.frame)
            elapsed = perf_counter() - start
            representative[f"{name}_score"] = scores
            representative[f"{name}_alert"] = (
                scores > thresholds[name][str(cfg["primary_alert_rate"])]
            )
            base = {
                "scenario": sid,
                "suite": suite,
                "family": family,
                "severity": severity,
                "injection_target": rate,
                "injection_seed": seed,
                "method": name,
                "fingerprint": trial.fingerprint,
                "score_seconds": elapsed,
            }
            for alert_rate in cfg["alert_rates"]:
                threshold = thresholds[name][str(alert_rate)]
                records.append(
                    {
                        **base,
                        "calibration_target": alert_rate,
                        "threshold": threshold,
                        **metrics(trial.labels, scores, threshold, test.date_time, trial.event_ids),
                    }
                )
            if suite == "primary":
                # Report actual sample sizes; small subgroups are descriptive, not definitive.
                groupings = {
                    "month": test.date_time.dt.strftime("%Y-%m"),
                    "day_type": np.where(test.date_time.dt.dayofweek < 5, "weekday", "weekend"),
                    "hour": test.date_time.dt.hour.astype(str),
                }
                for kind, groups in groupings.items():
                    for group in np.unique(groups):
                        mask = np.asarray(groups == group)
                        subgroup_records.append(
                            {
                                **base,
                                "group_kind": kind,
                                "group": group,
                                **metrics(
                                    trial.labels[mask],
                                    scores[mask],
                                    thresholds[name][str(cfg["primary_alert_rate"])],
                                ),
                            }
                        )
        if suite == "primary" and severity == "severe" and seed == cfg["injection_seeds"][0]:
            representative.to_csv(RESULTS / f"example_{family}.csv", index=False)
    # Isolate model randomness: same severe injected records, different fitted forest seeds.
    for model_seed in cfg["model_seeds"]:
        model = ForestDetector(cfg["n_estimators"], cfg["max_samples"], model_seed).fit(train)
        threshold = calibrate(model.score(val), cfg["primary_alert_rate"])
        for family, trial in trial_map.items():
            sensitivity.append(
                {
                    "model_seed": model_seed,
                    "family": family,
                    "threshold": threshold,
                    **metrics(
                        trial.labels,
                        model.score(trial.frame),
                        threshold,
                        test.date_time,
                        trial.event_ids,
                    ),
                }
            )
    results = pd.DataFrame(records)
    results.to_csv(RESULTS / "metrics.csv", index=False)
    pd.concat(changes, ignore_index=True).to_csv(
        RESULTS / "injected_changes.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    pd.concat(events, ignore_index=True).to_csv(RESULTS / "injected_events.csv", index=False)
    pd.DataFrame(manifests).to_csv(RESULTS / "scenario_manifest.csv", index=False)
    pd.DataFrame(subgroup_records).to_csv(RESULTS / "subgroups.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(RESULTS / "model_seed_sensitivity.csv", index=False)
    primary = results[
        (results.suite == "primary") & (results.calibration_target == cfg["primary_alert_rate"])
    ]
    summary = (
        primary.groupby(["family", "severity", "method"], sort=False)
        .agg(
            recall_median=("recall", "median"),
            recall_min=("recall", "min"),
            recall_max=("recall", "max"),
            precision_median=("precision", "median"),
            f1_median=("f1", "median"),
            ap_median=("average_precision", "median"),
            background_alert_rate_median=("background_alert_rate", "median"),
            event_recall_median=("event_recall", "median"),
            n_runs=("recall", "size"),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS / "summary.csv", index=False)
    pairs = primary.pivot(
        index=["family", "severity", "injection_seed"], columns="method", values="recall"
    ).reset_index()
    pairs["forest_minus_baseline"] = pairs.isolation_forest - pairs.typical_hour
    pairs.to_csv(RESULTS / "paired_recall_differences.csv", index=False)
    write_json(
        RESULTS / "run_manifest.json",
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "packages": {
                p: version(p) for p in ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]
            },
            "protocol_sha256": sha256(ARTIFACTS / "frozen_protocol.json"),
            "protocol_frozen_at_utc": frozen["frozen_at_utc"],
            "raw_source_sha256": json.loads((RESULTS / "data_audit.json").read_text())[
                "data_sha256"
            ],
            "n_test_scenarios": len(manifests),
            "n_metric_rows": len(records),
            "n_seed_sensitivity_runs": len(sensitivity),
            "result_sha256": {
                p.name: sha256(p)
                for p in sorted(RESULTS.iterdir())
                if p.name != "monthly_background_alerts.csv"
                and (p.suffix == ".csv" or p.name.endswith(".csv.gz"))
            },
        },
    )
    print(f"Completed {len(manifests)} paired test scenarios; saved {len(records)} metric rows.")
    return summary
