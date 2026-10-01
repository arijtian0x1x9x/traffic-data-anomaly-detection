"""Both detectors return a score where larger means more unusual."""

from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .data import load_split
from .features import make_features
from .paths import ARTIFACTS, PROCESSED, RESULTS, ROOT, config, prepare_dirs, sha256, write_json


class TypicalHour:
    """Absolute deviation from weekday/hour median, divided by robust spread."""

    def __init__(self, min_group=10, scale_floor=1.0):
        self.min_group = min_group
        self.scale_floor = scale_floor

    @staticmethod
    def _summary(values):
        median = float(np.median(values))
        return median, float(1.4826 * np.median(np.abs(np.asarray(values) - median)))

    def fit(self, frame):
        data = frame.copy()
        data["hour"] = data.date_time.dt.hour
        data["weekday"] = data.date_time.dt.dayofweek
        self.global_ = self._summary(data.traffic_volume)
        self.hours_ = {
            int(k): self._summary(v.traffic_volume)
            for k, v in data.groupby("hour")
            if len(v) >= self.min_group
        }
        self.groups_ = {
            tuple(k): self._summary(v.traffic_volume)
            for k, v in data.groupby(["weekday", "hour"])
            if len(v) >= self.min_group
        }
        return self

    def expected(self, frame):
        if not hasattr(self, "global_"):
            raise RuntimeError("Fit the baseline on training data first.")
        centres, scales = [], []
        for ts in frame.date_time:
            hour_group = self.hours_.get(ts.hour, self.global_)
            median, spread = self.groups_.get((ts.dayofweek, ts.hour), hour_group)
            # Sparse groups use broader statistics; zero spreads use broader positive spreads.
            if spread < self.scale_floor:
                spread = next(
                    (s for s in [hour_group[1], self.global_[1]] if s >= self.scale_floor),
                    self.scale_floor,
                )
            centres.append(median)
            scales.append(max(spread, self.scale_floor))
        return np.asarray(centres), np.asarray(scales)

    def score(self, frame):
        centres, scales = self.expected(frame)
        return np.abs(frame.traffic_volume.to_numpy() - centres) / scales


class ForestDetector:
    def __init__(self, n_estimators=200, max_samples="auto", seed=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination="auto",
            random_state=seed,
            n_jobs=1,
        )

    def fit(self, frame):
        self.model.fit(make_features(frame))
        return self

    def score(self, frame):
        return -self.model.score_samples(make_features(frame))


def calibrate(scores, alert_rate):
    """Quantile learned on validation only; strict > handles tied scores conservatively."""
    scores = np.asarray(scores, dtype=float)
    if not 0 < alert_rate < 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Calibration needs finite scores and a rate strictly between 0 and 1.")
    return float(np.quantile(scores, 1 - alert_rate, method="higher"))


def train():
    prepare_dirs()
    cfg = config()
    train_frame, val = load_split("train"), load_split("validation")
    models = {
        "typical_hour": TypicalHour(cfg["baseline_min_group"], cfg["baseline_scale_floor"]),
        "isolation_forest": ForestDetector(
            cfg["n_estimators"], cfg["max_samples"], cfg["model_seed"]
        ),
    }
    thresholds, records = {}, []
    for name, model in models.items():
        start = perf_counter()
        model.fit(train_frame)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        scores = model.score(val)
        score_seconds = perf_counter() - start
        thresholds[name] = {}
        for rate in cfg["alert_rates"]:
            threshold = calibrate(scores, rate)
            thresholds[name][str(rate)] = threshold
            records.append(
                {
                    "method": name,
                    "calibration_target": rate,
                    "threshold": threshold,
                    "validation_background_alert_rate": float(np.mean(scores > threshold)),
                    "fit_seconds": fit_seconds,
                    "validation_score_seconds": score_seconds,
                }
            )
        joblib.dump(model, ARTIFACTS / f"{name}.joblib")
    pd.DataFrame(records).to_csv(RESULTS / "calibration.csv", index=False)
    write_json(ARTIFACTS / "thresholds.json", thresholds)
    write_json(
        ARTIFACTS / "training_manifest.json",
        {
            "config_sha256": sha256(ROOT / "configs/study.json"),
            "train_sha256": sha256(PROCESSED / "train.csv"),
            "validation_sha256": sha256(PROCESSED / "validation.csv"),
            "model_sha256": {name: sha256(ARTIFACTS / f"{name}.joblib") for name in models},
            "thresholds_sha256": sha256(ARTIFACTS / "thresholds.json"),
            "n_train": len(train_frame),
            "n_validation": len(val),
            "features": list(make_features(val).columns),
            "baseline_scale": "1.4826 times MAD; broader-group fallback; minimum 1 vehicle",
            "score_direction": "higher means more unusual",
        },
    )
    print("Both detectors fitted on training only; thresholds calibrated on validation only.")
    return records


def load_models():
    import json

    path = ARTIFACTS / "training_manifest.json"
    if not path.exists():
        raise FileNotFoundError("Run 'uv run traffic-study train' first.")
    manifest = json.loads(path.read_text())
    checks = {
        "config_sha256": ROOT / "configs/study.json",
        "train_sha256": PROCESSED / "train.csv",
        "validation_sha256": PROCESSED / "validation.csv",
        "thresholds_sha256": ARTIFACTS / "thresholds.json",
    }
    if any(sha256(p) != manifest[k] for k, p in checks.items()):
        raise ValueError("Configuration/data changed since training. Re-run train and validate.")
    models = {}
    for name, expected in manifest["model_sha256"].items():
        path = ARTIFACTS / f"{name}.joblib"
        if sha256(path) != expected:
            raise ValueError("Saved model differs from its training manifest.")
        # These are trusted local artifacts generated by this code, never downloaded models.
        models[name] = joblib.load(path)
    return models, json.loads((ARTIFACTS / "thresholds.json").read_text())
