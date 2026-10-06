"""Chronology-aware train / validation / test splits.

The notebook used ``train_test_split(shuffle=True)`` on hourly samples. Neighbouring hours are
almost identical (autocorrelated weather and output), so a shuffled validation set contains
near-copies of training samples and over-states accuracy. The default here ("blocked") holds out
whole weeks and removes training hours adjacent to them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from renewcast.config import SplitConfig


@dataclass(frozen=True)
class Splits:
    """Integer positions into the aligned sample arrays."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _hours(times: pd.DatetimeIndex) -> np.ndarray:
    return times.values.astype("datetime64[h]").astype(np.int64)


def _drop_near(train: np.ndarray, val: np.ndarray, hours: np.ndarray, gap: int) -> np.ndarray:
    """Remove training samples within ``gap`` hours of any validation sample."""
    if gap <= 0 or len(val) == 0 or len(train) == 0:
        return train
    val_hours = np.sort(hours[val])
    train_hours = hours[train]
    pos = np.searchsorted(val_hours, train_hours)
    left = np.abs(train_hours - val_hours[np.clip(pos - 1, 0, len(val_hours) - 1)])
    right = np.abs(val_hours[np.clip(pos, 0, len(val_hours) - 1)] - train_hours)
    return train[np.minimum(left, right) > gap]


def make_splits(times: pd.DatetimeIndex, cfg: SplitConfig) -> Splits:
    years = times.year.to_numpy()
    pool = np.flatnonzero(np.isin(years, cfg.train_years))
    test = np.flatnonzero(np.isin(years, cfg.test_years))
    if len(pool) == 0:
        raise ValueError(f"No samples in train_years {cfg.train_years}")
    if len(test) == 0:
        raise ValueError(f"No samples in test_years {cfg.test_years}")

    hours = _hours(times)
    rng = np.random.default_rng(cfg.seed)
    n_val = max(1, round(cfg.val_fraction * len(pool)))

    if cfg.val_strategy == "random":
        shuffled = rng.permutation(pool)
        val, train = np.sort(shuffled[:n_val]), np.sort(shuffled[n_val:])
    elif cfg.val_strategy == "tail":
        val, train = pool[-n_val:], _drop_near(pool[:-n_val], pool[-n_val:], hours, cfg.gap_hours)
    else:  # blocked
        pool_hours = hours[pool]
        block_id = (pool_hours - pool_hours.min()) // cfg.block_hours
        ids = np.unique(block_id)
        if len(ids) < 2:
            raise ValueError("Training period is shorter than two blocks; reduce block_hours")
        n_blocks = min(max(1, round(cfg.val_fraction * len(ids))), len(ids) - 1)
        chosen = rng.choice(ids, size=n_blocks, replace=False)
        in_val = np.isin(block_id, chosen)
        val = pool[in_val]
        train = _drop_near(pool[~in_val], val, hours, cfg.gap_hours)

    if len(train) == 0 or len(val) == 0:
        raise ValueError(f"Empty split produced: {len(train)} train / {len(val)} val samples")
    return Splits(train=train, val=val, test=test)
