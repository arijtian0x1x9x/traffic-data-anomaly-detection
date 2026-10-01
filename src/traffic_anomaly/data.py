"""Download once, audit timestamps, and create strictly chronological splits."""

import io
import zipfile
from datetime import datetime, timezone
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .paths import PROCESSED, RAW, RESULTS, config, prepare_dirs, sha256, write_json

DATA_FILE = "Metro_Interstate_Traffic_Volume.csv.gz"


def download():
    """Retain the original UCI archive; never silently replace an existing download."""
    prepare_dirs()
    cfg = config()
    archive = RAW / "uci_492.zip"
    target = RAW / DATA_FILE
    if not archive.exists():
        with urlopen(cfg["dataset_url"], timeout=60) as response:
            payload = response.read()
        # Verify before writing; an HTML error page is not a dataset.
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if DATA_FILE not in z.namelist():
                raise ValueError("The UCI archive did not contain the expected dataset.")
        archive.write_bytes(payload)
    if not target.exists():
        with zipfile.ZipFile(archive) as z:
            target.write_bytes(z.read(DATA_FILE))
    metadata = RAW / "source.json"
    current_hash = sha256(target)
    if metadata.exists():
        import json

        if json.loads(metadata.read_text())["data_sha256"] != current_hash:
            raise ValueError("Raw dataset differs from source.json; investigate before continuing.")
    else:
        write_json(
            metadata,
            {
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "url": cfg["dataset_url"],
                "source_page": cfg["source_page"],
                "archive_sha256": sha256(archive),
                "data_sha256": current_hash,
                "citation": "Hogue, J. (2019). Metro Interstate Traffic Volume. UCI. DOI: 10.24432/C5X60B.",
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
            },
        )
    return target


def clean_hourly(raw):
    """Keep observed counts; exclude ambiguous conflicting timestamp groups."""
    df = raw.copy()
    required = {"date_time", "traffic_volume"}
    if not required <= set(df):
        raise ValueError(f"Missing columns: {required - set(df)}")
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["traffic_volume"] = pd.to_numeric(df["traffic_volume"], errors="coerce")
    valid = (
        df.date_time.notna()
        & np.isfinite(df.traffic_volume)
        & (df.traffic_volume >= 0)
        & (df.traffic_volume % 1 == 0)
    )
    valid &= df.date_time.eq(df.date_time.dt.floor("h"))
    good = df.loc[valid].copy()
    counts = good.groupby("date_time").traffic_volume.nunique()
    conflicts = counts[counts > 1].index
    conflict_rows = good[good.date_time.isin(conflicts)].copy()
    good = good[~good.date_time.isin(conflicts)]
    hourly = (
        good[["date_time", "traffic_volume"]]
        .drop_duplicates("date_time")
        .sort_values("date_time")
        .reset_index(drop=True)
    )
    hourly["traffic_volume"] = hourly.traffic_volume.astype("int64")
    if len(hourly) < 10:
        raise ValueError("Too few valid hourly observations.")
    full_range = pd.date_range(hourly.date_time.min(), hourly.date_time.max(), freq="h")
    missing = full_range.difference(hourly.date_time)
    gaps = hourly.date_time.diff().dt.total_seconds().div(3600).fillna(1)
    audit = {
        "raw_rows": int(len(raw)),
        "raw_columns": list(raw.columns),
        "raw_null_cells_by_column": {c: int(raw[c].isna().sum()) for c in raw},
        "invalid_rows_excluded": int((~valid).sum()),
        "repeated_timestamp_rows": int(df.loc[valid, "date_time"].duplicated().sum()),
        "conflicting_timestamps_excluded": int(len(conflicts)),
        "conflicting_rows_excluded": int(len(conflict_rows)),
        "same_count_duplicates_collapsed": int(len(good) - len(hourly)),
        "hourly_rows": int(len(hourly)),
        "first_timestamp": str(hourly.date_time.min()),
        "last_timestamp": str(hourly.date_time.max()),
        "missing_hours_in_full_span": int(len(missing)),
        "gap_count": int((gaps > 1).sum()),
        "longest_gap_missing_hours": int(max(gaps.max() - 1, 0)),
        "zero_volume_hours_retained": int((hourly.traffic_volume == 0).sum()),
    }
    return hourly, audit, conflict_rows, pd.DataFrame({"date_time": missing})


def chronological_split(hourly, train_fraction=0.6, validation_fraction=0.2):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("A non-empty test fraction is required.")
    if hourly.date_time.duplicated().any() or not hourly.date_time.is_monotonic_increasing:
        raise ValueError("Sort and deduplicate timestamps before splitting.")
    n = len(hourly)
    a, b = int(n * train_fraction), int(n * (train_fraction + validation_fraction))
    splits = {
        "train": hourly.iloc[:a].copy(),
        "validation": hourly.iloc[a:b].copy(),
        "test": hourly.iloc[b:].copy(),
    }
    if min(map(len, splits.values())) == 0:
        raise ValueError("Each split must contain observations.")
    return {k: v.reset_index(drop=True) for k, v in splits.items()}


def prepare():
    source = download()
    # "None" is a real holiday category, not a missing value.
    raw = pd.read_csv(source, keep_default_na=False)
    hourly, audit, conflicts, missing = clean_hourly(raw)
    cfg = config()
    splits = chronological_split(hourly, cfg["train_fraction"], cfg["validation_fraction"])
    hourly.to_csv(PROCESSED / "hourly.csv", index=False)
    for name, frame in splits.items():
        frame.to_csv(PROCESSED / f"{name}.csv", index=False)
    conflicts.to_csv(PROCESSED / "excluded_conflicts.csv", index=False)
    missing.to_csv(PROCESSED / "missing_hours.csv", index=False)
    audit["split_summary"] = {
        k: {"rows": len(v), "start": str(v.date_time.min()), "end": str(v.date_time.max())}
        for k, v in splits.items()
    }
    audit["rules"] = {
        k: cfg[k] for k in ["timestamp_policy", "duplicate_policy", "missing_hour_policy"]
    }
    audit["data_sha256"] = sha256(source)
    write_json(RESULTS / "data_audit.json", audit)
    print(f"Data ready: {len(raw):,} raw rows -> {len(hourly):,} observed unique hours.")
    return audit


def load_split(name):
    path = PROCESSED / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError("Run 'uv run traffic-study prepare' first.")
    return pd.read_csv(path, parse_dates=["date_time"])
