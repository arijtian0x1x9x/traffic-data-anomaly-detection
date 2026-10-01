"""Reproducible changes with exact labels and event bookkeeping."""

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOG_COLUMNS = [
    "position",
    "date_time",
    "original_value",
    "altered_value",
    "family",
    "severity",
    "event_id",
]
EVENT_COLUMNS = [
    "event_id",
    "family",
    "severity",
    "window_start",
    "window_end",
    "first_changed",
    "last_changed",
    "changed_hours",
]


@dataclass
class Injection:
    frame: pd.DataFrame
    labels: np.ndarray
    event_ids: np.ndarray
    changes: pd.DataFrame
    events: pd.DataFrame
    eligible_hours: int
    fingerprint: str


def _candidates(frame, length, require_positive):
    values = frame.traffic_volume.to_numpy()
    delta = frame.date_time.diff().dt.total_seconds().to_numpy() / 3600
    candidates = []
    for start in range(len(frame) - length + 1):
        stop = start + length
        if length > 1 and not np.all(delta[start + 1 : stop] == 1):
            continue
        if require_positive and not np.all(values[start:stop] > 0):
            continue
        if not require_positive and length > 1 and np.all(values[start:stop] == values[start]):
            continue
        candidates.append(start)
    return np.asarray(candidates, dtype=int)


def inject(frame, family, severity, rate, seed, cfg):
    """Alter counts only, never timestamps. No event crosses a missing-hour gap."""
    if family not in [*cfg["families"], "mixed"] or severity not in cfg["severities"]:
        raise ValueError("Unknown anomaly family or severity.")
    if not 0 < rate < 1:
        raise ValueError("Injection rate must be between zero and one.")
    if frame.date_time.duplicated().any() or not frame.date_time.is_monotonic_increasing:
        raise ValueError("Injection requires ordered unique timestamps.")
    original = frame.reset_index(drop=True).copy(deep=True)
    result = original.copy(deep=True)
    rng = np.random.default_rng(seed)
    level = cfg["severities"].index(severity)
    families = cfg["families"] if family == "mixed" else [family]
    lengths = {f: (1 if f in ["spike", "drop"] else cfg["event_lengths"][level]) for f in families}
    starts = {f: _candidates(original, lengths[f], f != "flatline") for f in families}
    eligible = np.zeros(len(original), dtype=bool)
    for f in families:
        for offset in range(lengths[f]):
            eligible[starts[f] + offset] = True
    target = max(1, int(np.ceil(eligible.sum() * rate)))
    occupied = np.zeros(len(original), dtype=bool)
    labels = np.zeros(len(original), dtype=np.int8)
    event_ids = np.full(len(original), -1, dtype=int)
    logs, events = [], []
    # Shuffle all candidate events once. This terminates even on tiny or constant inputs.
    options = [(f, int(i)) for f in families for i in starts[f]]
    rng.shuffle(options)
    original_values = original.traffic_volume.to_numpy(dtype=np.int64)
    values = original_values.copy()
    for f, start in options:
        stop = start + lengths[f]
        if occupied[start:stop].any():
            continue
        old = original_values[start:stop]
        if f == "spike":
            new = np.rint(old * cfg["spike_factors"][level]).astype(np.int64)
        elif f == "drop":
            new = np.rint(old * cfg["drop_factors"][level]).astype(np.int64)
        elif f == "shift":
            direction = rng.choice([-1, 1])
            new = np.rint(old * (1 + direction * cfg["shift_magnitudes"][level])).astype(np.int64)
        else:
            new = np.full_like(old, old[0])
        new = np.maximum(0, new)
        positions = np.arange(start, stop)[new != old]
        if len(positions) == 0:
            continue
        eid = len(events)
        occupied[start:stop] = True
        values[start:stop] = new
        labels[positions], event_ids[positions] = 1, eid
        for pos in positions:
            logs.append(
                [
                    int(pos),
                    original.date_time.iloc[pos],
                    int(original_values[pos]),
                    int(values[pos]),
                    f,
                    severity,
                    eid,
                ]
            )
        events.append(
            [
                eid,
                f,
                severity,
                original.date_time.iloc[start],
                original.date_time.iloc[stop - 1],
                original.date_time.iloc[positions[0]],
                original.date_time.iloc[positions[-1]],
                int(len(positions)),
            ]
        )
        if labels.sum() >= target:
            break
    result["traffic_volume"] = values
    # A label means an actual numeric change, not merely a selected location.
    assert np.array_equal(labels.astype(bool), values != original_values)
    digest = hashlib.sha256()
    digest.update(original.date_time.astype("int64").to_numpy().tobytes())
    digest.update(values.tobytes())
    digest.update(labels.tobytes())
    return Injection(
        result,
        labels,
        event_ids,
        pd.DataFrame(logs, columns=LOG_COLUMNS),
        pd.DataFrame(events, columns=EVENT_COLUMNS),
        int(eligible.sum()),
        digest.hexdigest(),
    )
