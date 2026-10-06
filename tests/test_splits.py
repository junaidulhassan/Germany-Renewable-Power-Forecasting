from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from renewcast.config import SplitConfig
from renewcast.data.splits import make_splits


@pytest.fixture
def times() -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", "2024-06-30 23:00", freq="h")


def _cfg(**kwargs) -> SplitConfig:
    return SplitConfig(train_years=[2023], test_years=[2024], **kwargs)


@pytest.mark.parametrize("strategy", ["blocked", "tail", "random"])
def test_splits_are_disjoint_and_respect_years(times: pd.DatetimeIndex, strategy: str) -> None:
    s = make_splits(times, _cfg(val_strategy=strategy))
    train, val, test = set(s.train), set(s.val), set(s.test)
    assert not (train & val) and not (train & test) and not (val & test)
    assert set(times[s.train].year) == {2023}
    assert set(times[s.val].year) == {2023}
    assert set(times[s.test].year) == {2024}
    assert len(s.test) == (times.year == 2024).sum()


def test_validation_fraction_is_approximately_respected(times: pd.DatetimeIndex) -> None:
    pool = int((times.year == 2023).sum())
    for strategy in ("blocked", "tail", "random"):
        s = make_splits(times, _cfg(val_strategy=strategy, val_fraction=0.1))
        assert abs(len(s.val) / pool - 0.1) < 0.03, strategy


def test_blocked_validation_consists_of_contiguous_blocks(times: pd.DatetimeIndex) -> None:
    s = make_splits(times, _cfg(val_strategy="blocked", block_hours=168))
    steps = np.diff(times[s.val].values.astype("datetime64[h]").astype(int))
    # Mostly 1-hour steps inside a block; a few large jumps between blocks.
    assert (steps == 1).mean() > 0.95


def test_gap_separates_training_from_validation(times: pd.DatetimeIndex) -> None:
    for strategy in ("blocked", "tail"):
        s = make_splits(times, _cfg(val_strategy=strategy, gap_hours=24))
        train_h = times[s.train].values.astype("datetime64[h]").astype(int)
        val_h = np.sort(times[s.val].values.astype("datetime64[h]").astype(int))
        nearest = np.min(np.abs(train_h[:, None] - val_h[None, :]), axis=1)
        assert nearest.min() > 24, strategy


def test_random_strategy_has_no_gap_and_interleaves(times: pd.DatetimeIndex) -> None:
    s = make_splits(times, _cfg(val_strategy="random"))
    assert len(s.train) + len(s.val) == (times.year == 2023).sum()
    assert np.abs(np.diff(s.val)).min() >= 1
    assert (np.diff(s.val) == 1).mean() < 0.5  # shuffled, not contiguous


def test_splits_are_deterministic_for_a_seed(times: pd.DatetimeIndex) -> None:
    a = make_splits(times, _cfg(seed=7))
    b = make_splits(times, _cfg(seed=7))
    c = make_splits(times, _cfg(seed=8))
    assert np.array_equal(a.val, b.val)
    assert not np.array_equal(a.val, c.val)


def test_missing_period_raises(times: pd.DatetimeIndex) -> None:
    with pytest.raises(ValueError, match="test_years"):
        make_splits(times, SplitConfig(train_years=[2023], test_years=[2030]))
    with pytest.raises(ValueError, match="train_years"):
        make_splits(times, SplitConfig(train_years=[2019], test_years=[2024]))


def test_train_period_shorter_than_two_blocks_raises() -> None:
    short = pd.date_range("2023-01-01", periods=100, freq="h").append(
        pd.date_range("2024-01-01", periods=10, freq="h")
    )
    with pytest.raises(ValueError, match="two blocks"):
        make_splits(short, _cfg(val_strategy="blocked", block_hours=168))
