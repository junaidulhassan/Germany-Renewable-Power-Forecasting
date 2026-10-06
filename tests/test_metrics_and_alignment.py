from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from renewcast.data.prepare import align_samples
from renewcast.evaluation.metrics import monthly_report, regression_report


def test_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    report = regression_report(y, y)
    assert report["r2"] == pytest.approx(1.0)
    assert report["mae"] == 0.0 and report["rmse"] == 0.0 and report["bias"] == 0.0


def test_known_values() -> None:
    y = np.array([0.0, 10.0, 20.0, 30.0])
    p = np.array([5.0, 10.0, 15.0, 40.0])
    report = regression_report(y, p)
    assert report["mae"] == pytest.approx(5.0)
    assert report["rmse"] == pytest.approx(np.sqrt((25 + 0 + 25 + 100) / 4))
    assert report["bias"] == pytest.approx(2.5)
    assert report["r2"] == pytest.approx(1 - 150 / 500)
    assert report["nmae"] == pytest.approx(5.0 / 15.0)


def test_predicting_the_mean_gives_zero_r2() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert regression_report(y, np.full(4, y.mean()))["r2"] == pytest.approx(0.0)


def test_metrics_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        regression_report(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        regression_report(np.array([]), np.array([]))


def test_monthly_report_has_one_row_per_month() -> None:
    times = pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h")
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 100, len(times))
    out = monthly_report(times, y, y + 1.0)
    assert out["month"].tolist() == ["2024-01", "2024-02", "2024-03"]
    assert np.allclose(out["mae"], 1.0)


def test_align_inner_joins_on_timestamp_and_drops_nan() -> None:
    cube_times = pd.date_range("2023-01-01", periods=10, freq="h")
    target = pd.Series(
        np.arange(12, dtype=float), index=pd.date_range("2023-01-01 02:00", periods=12, freq="h")
    )
    target.iloc[1] = np.nan  # 03:00 has no target value
    times, rows, y = align_samples(cube_times, target, [2023])

    assert list(times) == [
        t
        for t in cube_times
        if t >= pd.Timestamp("2023-01-01 02:00") and t != pd.Timestamp("2023-01-01 03:00")
    ]
    assert (cube_times[rows] == times).all()
    assert y[0] == 0.0 and y[1] == 2.0  # target at 02:00 and 04:00


def test_align_filters_years_and_raises_without_overlap() -> None:
    cube_times = pd.date_range("2023-12-31 20:00", periods=8, freq="h")  # spans the new year
    target = pd.Series(1.0, index=cube_times)
    times, _, _ = align_samples(cube_times, target, [2024])
    assert times.year.unique().tolist() == [2024]
    with pytest.raises(ValueError, match="No overlapping"):
        align_samples(cube_times, target, [2030])
