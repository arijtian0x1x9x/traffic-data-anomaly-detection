"""Metrics respect synthetic labels and retain all hours of multi-hour events."""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def metrics(labels, scores, threshold, timestamps=None, event_ids=None):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Labels and finite scores must have equal lengths.")
    pred = scores > threshold
    pos, neg = labels == 1, labels == 0
    values = {
        "n_hours": len(labels),
        "n_injected": int(pos.sum()),
        "n_background": int(neg.sum()),
        "true_positive": int((pred & pos).sum()),
        "false_negative": int((~pred & pos).sum()),
        "benchmark_false_positive": int((pred & neg).sum()),
        "true_negative": int((~pred & neg).sum()),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)) if pos.any() else None,
        "background_alert_rate": float(pred[neg].mean()) if neg.any() else None,
        "alert_rate": float(pred.mean()),
        "actual_injection_rate": float(pos.mean()),
    }
    if event_ids is not None:
        ids = np.asarray(event_ids)
        unique = np.unique(ids[ids >= 0])
        detected, delays = 0, []
        for eid in unique:
            indexes = np.flatnonzero(ids == eid)
            hits = indexes[pred[indexes]]
            if len(hits):
                detected += 1
                delays.append(
                    (timestamps.iloc[hits[0]] - timestamps.iloc[indexes[0]]).total_seconds() / 3600
                )
        values.update(
            {
                "n_events": len(unique),
                "events_detected": detected,
                "event_recall": detected / len(unique) if len(unique) else None,
                "median_delay_hours_detected_only": float(np.median(delays)) if delays else None,
            }
        )
    return values
