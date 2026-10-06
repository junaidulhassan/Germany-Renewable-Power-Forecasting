from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from renewcast.data.generation import impute_missing, localize_to_utc, to_hourly


def _quarter_hours(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="15min")


def test_summer_time_is_converted_to_utc() -> None:
    df = pd.DataFrame({"MTU": ["2023-07-01 12:00:00"], "solar": [1.0]})
    out = localize_to_utc(df, "MTU", "Europe/Berlin")
    assert out.index[0] == pd.Timestamp("2023-07-01 10:00:00")  # CEST = UTC+2


def test_winter_time_is_converted_to_utc() -> None:
    df = pd.DataFrame({"MTU": ["2023-01-01 12:00:00"], "solar": [1.0]})
    out = localize_to_utc(df, "MTU", "Europe/Berlin")
    assert out.index[0] == pd.Timestamp("2023-01-01 11:00:00")  # CET = UTC+1


def test_autumn_dst_repeated_hour_becomes_unique_utc() -> None:
    # Clocks go back on 2023-10-29: local 02:00-02:59 occurs twice in the ENTSO-E export.
    stamps = list(_quarter_hours("2023-10-29 01:00", "2023-10-29 02:45"))
    stamps += list(_quarter_hours("2023-10-29 02:00", "2023-10-29 04:00"))
    df = pd.DataFrame({"MTU": stamps, "solar": np.arange(len(stamps), dtype=float)})
    out = localize_to_utc(df, "MTU", "Europe/Berlin")

    assert out.index.is_unique
    assert out.index.is_monotonic_increasing
    # Continuous 15-minute grid in UTC: no gaps and no overlaps.
    assert (out.index.to_series().diff().dropna() == pd.Timedelta("15min")).all()
    # First 02:00 (summer time) is 00:00 UTC, second 02:00 (winter time) is 01:00 UTC.
    assert out.loc["2023-10-29 00:00:00", "solar"] == 4.0
    assert out.loc["2023-10-29 01:00:00", "solar"] == 8.0


def test_spring_dst_gap_has_no_duplicates() -> None:
    stamps = list(_quarter_hours("2023-03-26 00:00", "2023-03-26 01:45"))
    stamps += list(_quarter_hours("2023-03-26 03:00", "2023-03-26 04:00"))
    df = pd.DataFrame({"MTU": stamps, "solar": 1.0})
    out = localize_to_utc(df, "MTU", "Europe/Berlin")
    assert out.index.is_unique
    assert (out.index.to_series().diff().dropna() == pd.Timedelta("15min")).all()


def test_utc_source_is_left_untouched() -> None:
    df = pd.DataFrame({"MTU": ["2023-07-01 12:00:00"], "solar": [1.0]})
    out = localize_to_utc(df, "MTU", "UTC")
    assert out.index[0] == pd.Timestamp("2023-07-01 12:00:00")


def test_interpolate_fills_gaps_using_time() -> None:
    index = pd.DatetimeIndex(_quarter_hours("2023-01-01 00:00", "2023-01-01 01:00"))
    df = pd.DataFrame({"a": [0.0, np.nan, np.nan, 3.0, 4.0]}, index=index)
    out = impute_missing(df, "interpolate")
    assert out["a"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_impute_none_keeps_nan() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan]}, index=pd.date_range("2023-01-01", periods=2, freq="h"))
    assert impute_missing(df, "none")["a"].isna().sum() == 1


def test_iterative_imputer_leaves_no_nan() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(size=(50, 3)),
        columns=list("abc"),
        index=pd.date_range("2023-01-01", periods=50, freq="h"),
    )
    df.iloc[3, 0] = np.nan
    assert not impute_missing(df, "iterative").isna().any().any()


def test_hourly_instant_vs_mean() -> None:
    index = pd.DatetimeIndex(_quarter_hours("2023-01-01 00:00", "2023-01-01 01:45"))
    df = pd.DataFrame({"a": [0.0, 4.0, 8.0, 12.0, 100.0, 100.0, 100.0, 100.0]}, index=index)
    instant = to_hourly(df, "instant")
    mean = to_hourly(df, "mean")
    assert instant["a"].tolist() == [0.0, 100.0]
    assert mean["a"].tolist() == [6.0, 100.0]
    assert list(mean.index) == list(instant.index)


def test_unknown_methods_raise() -> None:
    df = pd.DataFrame({"a": [np.nan]}, index=pd.date_range("2023-01-01", periods=1, freq="h"))
    with pytest.raises(ValueError):
        impute_missing(df, "magic")
    with pytest.raises(ValueError):
        to_hourly(df, "magic")
