import numpy as np
import pandas as pd
import pytest

from traffic_anomaly.data import chronological_split, clean_hourly
from traffic_anomaly.features import FEATURE_NAMES, make_features
from traffic_anomaly.models import ForestDetector, TypicalHour, calibrate


def hourly(n=1000):
    return pd.DataFrame(
        {
            "date_time": pd.date_range("2020-01-01", periods=n, freq="h"),
            "traffic_volume": 1000 + np.arange(n) % 24 * 100,
        }
    )


def test_duplicate_counts_are_not_summed_and_conflicts_are_excluded():
    base = hourly(20)
    raw = pd.concat([base, base.iloc[[0]], base.iloc[[1]].assign(traffic_volume=9999)])
    clean, audit, conflicts, _ = clean_hourly(raw)
    assert len(clean) == 19
    assert clean.iloc[0].traffic_volume == 1000
    assert base.date_time.iloc[1] not in set(clean.date_time)
    assert audit["same_count_duplicates_collapsed"] == 1
    assert audit["conflicting_timestamps_excluded"] == 1
    assert len(conflicts) == 2


def test_gaps_stay_missing_and_zero_is_retained():
    raw = hourly(20).drop(index=[5, 6])
    raw.loc[0, "traffic_volume"] = 0
    clean, audit, _, missing = clean_hourly(raw)
    assert len(clean) == 18 and len(missing) == 2
    assert audit["zero_volume_hours_retained"] == 1
    assert audit["longest_gap_missing_hours"] == 2


def test_chronological_split_is_disjoint_and_ordered():
    splits = chronological_split(hourly(100))
    a, b, c = [splits[s] for s in ["train", "validation", "test"]]
    assert (len(a), len(b), len(c)) == (60, 20, 20)
    assert a.date_time.max() < b.date_time.min() < c.date_time.min()
    assert b.date_time.max() < c.date_time.min()


def test_duplicate_and_unsorted_timestamps_rejected():
    frame = hourly(20)
    with pytest.raises(ValueError):
        chronological_split(frame.iloc[::-1])
    with pytest.raises(ValueError):
        chronological_split(pd.concat([frame, frame.iloc[[0]]]))


def test_negative_fraction_rejected():
    with pytest.raises(ValueError):
        chronological_split(hourly(20), -0.1, 0.2)


def test_invalid_counts_and_timestamps_are_excluded():
    frame = hourly(20)
    frame["traffic_volume"] = frame.traffic_volume.astype(float)
    frame.loc[0, "traffic_volume"] = -5
    frame.loc[1, "traffic_volume"] = np.nan
    frame.loc[2, "date_time"] = pd.NaT
    _, audit, _, _ = clean_hourly(frame)
    assert audit["invalid_rows_excluded"] == 3


def test_baseline_zero_spread_and_unseen_group_remain_finite():
    frame = hourly(30).assign(traffic_volume=100)
    model = TypicalHour(min_group=10).fit(frame)
    assert np.isfinite(model.score(hourly(100))).all()
    assert (model.expected(hourly(100))[1] > 0).all()


def test_baseline_is_unchanged_by_scoring_future_data():
    train, future = hourly(1000), hourly(24).assign(traffic_volume=99999)
    model = TypicalHour().fit(train)
    before = model.score(train).copy()
    model.score(future)
    np.testing.assert_array_equal(before, model.score(train))


def test_features_exclude_labels_and_original_unaltered_volume():
    frame = hourly(10).assign(injected=1, original_volume=-100, event_id=10)
    features = make_features(frame)
    assert list(features) == FEATURE_NAMES
    np.testing.assert_array_equal(features.traffic_volume, frame.traffic_volume)


def test_clock_encoding_wraps_midnight():
    x = make_features(
        pd.DataFrame(
            {
                "date_time": pd.to_datetime(
                    ["2020-01-01 23:00", "2020-01-02 00:00", "2020-01-02 12:00"]
                ),
                "traffic_volume": [100, 100, 100],
            }
        )
    )
    clock = x[["hour_sin", "hour_cos"]].to_numpy()
    assert np.linalg.norm(clock[0] - clock[1]) < np.linalg.norm(clock[1] - clock[2])


def test_calibration_ties_are_conservative():
    scores = np.array([1] * 99 + [10])
    t = calibrate(scores, 0.01)
    assert np.mean(scores > t) <= 0.01
    with pytest.raises(ValueError):
        calibrate([np.nan], 0.01)


def test_forest_fit_and_score_are_deterministic_for_same_seed():
    frame = hourly(600)
    one = ForestDetector(n_estimators=25, seed=42).fit(frame)
    two = ForestDetector(n_estimators=25, seed=42).fit(frame)
    np.testing.assert_array_equal(one.score(frame), two.score(frame))
