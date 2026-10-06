"""Build hourly generation targets from the ENTSO-E per-production-type CSV.

The raw file has 15-minute resolution and timestamps (``MTU``) in German local time, with a
repeated hour every October when clocks go back. Weather grids are in UTC, so timestamps are
converted to UTC before anything is joined.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from renewcast.config import Settings

logger = logging.getLogger(__name__)

TIME_INDEX_NAME = "time_utc"
TARGETS_FILE = "targets_hourly.csv"


def load_raw_generation(path: Path, timestamp_column: str, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Generation CSV not found at {path}. See data/README.md for where to obtain it."
        )
    return pd.read_csv(path, usecols=[timestamp_column, *columns])


def localize_to_utc(df: pd.DataFrame, timestamp_column: str, timezone: str) -> pd.DataFrame:
    """Parse local timestamps and return the frame indexed by naive UTC time.

    In the autumn DST transition the same wall-clock hour appears twice. The first occurrence is
    summer time, the second winter time; this holds for chronologically ordered ENTSO-E exports.
    """
    stamps = pd.to_datetime(df[timestamp_column])
    if timezone.upper() == "UTC":
        utc = stamps
    else:
        second_occurrence = stamps.duplicated(keep="first").to_numpy()
        local = stamps.dt.tz_localize(
            timezone,
            ambiguous=~second_occurrence,  # True marks the summer-time (first) occurrence
            nonexistent="shift_forward",
        )
        utc = local.dt.tz_convert("UTC").dt.tz_localize(None)

    out = df.drop(columns=[timestamp_column]).copy()
    out.index = pd.DatetimeIndex(utc, name=TIME_INDEX_NAME)
    out = out.sort_index()
    duplicated = out.index.duplicated(keep="first")
    if duplicated.any():
        logger.warning("Dropping %d duplicated UTC timestamps", int(duplicated.sum()))
        out = out[~duplicated]
    return out


def impute_missing(df: pd.DataFrame, method: str, seed: int = 42) -> pd.DataFrame:
    n_missing = int(df.isna().sum().sum())
    if method == "none" or n_missing == 0:
        logger.info("Imputation skipped (%d missing values)", n_missing)
        return df
    logger.info("Imputing %d missing values with '%s'", n_missing, method)
    if method == "interpolate":
        return df.interpolate(method="time", limit_direction="both")
    if method == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        imputer = IterativeImputer(max_iter=10, random_state=seed)
        out = df.copy()
        out[:] = imputer.fit_transform(df)
        return out
    raise ValueError(f"Unknown imputation method '{method}'")


def to_hourly(df: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "instant":
        return df[df.index.minute == 0]
    if how == "mean":
        return df.resample("60min").mean()
    raise ValueError(f"Unknown hourly aggregation '{how}'")


def build_targets(settings: Settings) -> pd.DataFrame:
    """Return an hourly UTC frame with one column per configured target (in MW)."""
    cfg = settings.generation
    columns = sorted({c for t in settings.targets.values() for c in t.generation_columns})
    raw = load_raw_generation(settings.paths.raw_generation, cfg.timestamp_column, columns)
    frame = localize_to_utc(raw, cfg.timestamp_column, cfg.source_timezone)
    frame = frame[frame.index.year.isin(settings.years)]
    frame = impute_missing(frame, cfg.imputation, seed=settings.runtime.seed)
    frame = to_hourly(frame, cfg.hourly_aggregation)

    targets = pd.DataFrame(index=frame.index)
    for name, target in settings.targets.items():
        targets[name] = frame[target.generation_columns].sum(axis=1, min_count=1)
    logger.info(
        "Built targets %s: %d hourly rows, %s -> %s",
        list(targets.columns),
        len(targets),
        targets.index.min(),
        targets.index.max(),
    )
    return targets


def count_missing(settings: Settings) -> pd.DataFrame:
    """Missing values per raw column within the configured years (input to the EDA bar chart)."""
    cfg = settings.generation
    columns = sorted({c for t in settings.targets.values() for c in t.generation_columns})
    raw = load_raw_generation(settings.paths.raw_generation, cfg.timestamp_column, columns)
    if settings.eda.missing_values_year_filter:
        years = pd.to_datetime(raw[cfg.timestamp_column]).dt.year
        raw = raw[years.isin(settings.years)]
    counts = raw.isna().sum().reset_index()
    counts.columns = ["Column", "Null Count"]
    return counts


def save_targets(df: pd.DataFrame, interim_dir: Path) -> Path:
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / TARGETS_FILE
    df.to_csv(path, index_label=TIME_INDEX_NAME)
    return path


def load_targets(interim_dir: Path) -> pd.DataFrame:
    path = interim_dir / TARGETS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `renewcast prepare-targets` first.")
    return pd.read_csv(path, index_col=TIME_INDEX_NAME, parse_dates=True)
