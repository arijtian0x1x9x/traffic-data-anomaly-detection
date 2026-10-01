import numpy as np
import pandas as pd
import pytest

from traffic_anomaly.evaluation import metrics
from traffic_anomaly.injection import inject
from traffic_anomaly.paths import config


def sample():
    ts = pd.date_range("2020-01-01", periods=1000, freq="h").delete(np.arange(200, 225))
    return pd.DataFrame({"date_time": ts, "traffic_volume": 1000 + np.arange(len(ts)) % 24 * 50})


@pytest.mark.parametrize("family", ["spike", "drop", "shift", "flatline", "mixed"])
def test_injection_exact_labels_reproducibility_and_gap_boundaries(family):
    frame, cfg = sample(), config()
    original = frame.copy(deep=True)
    trial = inject(frame, family, "severe", 0.05, 101, cfg)
    repeated = inject(frame, family, "severe", 0.05, 101, cfg)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(trial.frame, repeated.frame)
    assert trial.fingerprint == repeated.fingerprint
    assert trial.labels.sum() > 0
    np.testing.assert_array_equal(
        trial.labels.astype(bool), trial.frame.traffic_volume != frame.traffic_volume
    )
    assert (trial.frame.traffic_volume >= 0).all()
    assert len(trial.changes) == trial.labels.sum()
    assert not trial.changes.position.duplicated().any()
    assert (trial.event_ids[trial.labels == 0] == -1).all()
    for _, event in trial.events.iterrows():
        window = frame[frame.date_time.between(event.window_start, event.window_end)]
        assert (window.date_time.diff().dropna() == pd.Timedelta(hours=1)).all()
    # Restoring the change log exactly reconstructs the unmodified copy.
    restored = trial.frame.copy()
    restored.loc[trial.changes.position, "traffic_volume"] = trial.changes.original_value.to_numpy()
    pd.testing.assert_frame_equal(restored, frame)


def test_zero_counts_do_not_become_fake_multiplicative_anomalies():
    frame = sample().assign(traffic_volume=0)
    trial = inject(frame, "spike", "severe", 0.01, 101, config())
    assert trial.labels.sum() == 0 and trial.eligible_hours == 0


def test_constant_flatline_does_not_receive_positive_labels():
    trial = inject(sample().assign(traffic_volume=100), "flatline", "severe", 0.01, 1, config())
    assert trial.labels.sum() == 0


def test_metrics_known_confusion_matrix():
    result = metrics([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2], 0.5)
    assert result["true_positive"] == 1
    assert result["false_negative"] == 1
    assert result["benchmark_false_positive"] == 1
    assert result["recall"] == result["precision"] == result["f1"] == 0.5
    assert result["background_alert_rate"] == 0.5


def test_event_recall_does_not_inflate_hour_recall():
    times = pd.Series(pd.date_range("2020-01-01", periods=6, freq="h"))
    result = metrics([1, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0], 0.5, times, [0, 0, 0, 0, -1, -1])
    assert result["recall"] == 0.25
    assert result["event_recall"] == 1
    assert result["median_delay_hours_detected_only"] == 2


def test_no_positives_has_no_average_precision():
    result = metrics([0, 0], [1, 0], 0.5)
    assert result["average_precision"] is None
    assert result["background_alert_rate"] == 0.5


def test_scores_must_be_finite():
    with pytest.raises(ValueError):
        metrics([1, 0], [np.nan, 1], 0.5)
