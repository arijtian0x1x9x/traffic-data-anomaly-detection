"""Use only traffic count and calendar information available at the current hour."""

import numpy as np
import pandas as pd

FEATURE_NAMES = ["traffic_volume", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]


def make_features(frame):
    timestamp = pd.to_datetime(frame["date_time"])
    volume = pd.to_numeric(frame["traffic_volume"]).to_numpy(dtype=float)
    if not np.isfinite(volume).all() or timestamp.isna().any() or (volume < 0).any():
        raise ValueError("Features require valid timestamps and finite nonnegative counts.")
    hour, weekday = timestamp.dt.hour.to_numpy(), timestamp.dt.dayofweek.to_numpy()
    return pd.DataFrame(
        {
            "traffic_volume": volume,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
        },
        columns=FEATURE_NAMES,
    )
