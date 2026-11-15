import numpy as np
import pandas as pd

from commit_risk.train import bugs_caught, drop_immature_labels, temporal_split


def _dates(*dates):
    return pd.DataFrame({"date": list(dates), "introduced_bug": [0] * len(dates)})


def test_drop_immature_labels_is_relative_to_latest_commit():
    df = _dates("2025-01-01", "2025-06-01", "2025-12-01", "2026-01-01")

    kept = drop_immature_labels(df, days=180)

    assert list(kept["date"]) == ["2025-01-01", "2025-06-01"]


def test_temporal_split_never_puts_future_commits_in_training():
    df = _dates("2025-05-01", "2025-01-01", "2025-04-01", "2025-02-01", "2025-03-01")

    train, test = temporal_split(df, test_fraction=0.4)

    assert list(train["date"]) == ["2025-01-01", "2025-02-01", "2025-03-01"]
    assert list(test["date"]) == ["2025-04-01", "2025-05-01"]
    assert train["date"].max() < test["date"].min()


def test_bugs_caught_reviewing_top_ranked():
    y = pd.Series([1, 0, 1, 0])
    risk = np.array([0.9, 0.8, 0.1, 0.2])

    assert bugs_caught(y, risk, 0.25) == 0.5  # top 1 commit catches 1 of 2 bugs
    assert bugs_caught(y, risk, 1.0) == 1.0
