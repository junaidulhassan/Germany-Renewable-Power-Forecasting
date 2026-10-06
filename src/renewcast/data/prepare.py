"""Join weather frames with generation targets on UTC timestamps and split them."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from renewcast.config import Settings
from renewcast.data.generation import load_targets
from renewcast.data.splits import Splits, make_splits
from renewcast.data.weather import WeatherCube

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedData:
    """Hours that have both a weather frame and a target value, in chronological order."""

    cube: WeatherCube
    times: pd.DatetimeIndex  # UTC timestamp of each sample
    rows: np.ndarray  # index of each sample's frame inside ``cube.data``
    y: np.ndarray  # target in MW, float64
    splits: Splits


def align_samples(
    cube_times: pd.DatetimeIndex, target: pd.Series, years: list[int]
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Inner-join cube frames and target values on timestamp. Returns (times, rows, y)."""
    series = target.dropna()
    series = series[series.index.year.isin(years)]
    common = cube_times.intersection(series.index).sort_values()
    if len(common) == 0:
        raise ValueError("No overlapping timestamps between weather cube and target series")
    rows = cube_times.get_indexer(common)
    y = series.loc[common].to_numpy(dtype=np.float64)
    dropped = len(series) - len(common)
    if dropped:
        logger.warning("%d target hours have no weather frame and were dropped", dropped)
    return common, rows, y


def prepare_target_data(settings: Settings, target_name: str) -> PreparedData:
    target_cfg = settings.targets[target_name]
    cube = WeatherCube.load(settings.paths.processed_dir, target_cfg.weather_variable)
    targets = load_targets(settings.paths.interim_dir)
    if target_name not in targets:
        raise KeyError(f"Target '{target_name}' not in targets file; rerun `prepare-targets`")
    times, rows, y = align_samples(cube.times, targets[target_name], settings.years)
    splits = make_splits(times, settings.split)
    logger.info("%s: %d aligned hours, splits %s", target_name, len(times), splits.sizes())
    return PreparedData(cube=cube, times=times, rows=rows, y=y, splits=splits)
