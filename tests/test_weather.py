"""Builds tiny synthetic HOSTRADA-like NetCDF files and checks the memmap conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renewcast.config import Settings
from renewcast.data.download import select_files
from renewcast.data.weather import WeatherCube, find_nc_files, preprocess_variable
from tests.conftest import REPO_ROOT

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")


def _write_month(path: Path, start: str, hours: int, shape=(8, 6), value0: float = 0.0) -> None:
    times = pd.date_range(start, periods=hours, freq="h")
    data = (
        np.arange(hours, dtype=np.float32)[:, None, None]
        + value0
        + np.zeros((hours, *shape), np.float32)
    )
    data[:, 0, 0] = np.nan  # masked cell
    ds = xr.Dataset({"rsds": (("time", "Y", "X"), data)}, coords={"time": times})
    ds.to_netcdf(path)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    overrides = [
        f"paths.weather_raw_dir={tmp_path / 'raw'}",
        f"paths.processed_dir={tmp_path / 'processed'}",
        "weather.downsample_factor=2",
    ]
    s = Settings.load(REPO_ROOT / "configs" / "default.yaml", overrides)
    raw = tmp_path / "raw" / "rsds"
    raw.mkdir(parents=True)
    # written out of order on purpose; the pipeline must sort by month
    _write_month(
        raw / "rsds_1hr_HOSTRADA-v1-0_BE_gn_2023020100-2023020123.nc", "2023-02-01", 24, value0=100
    )
    _write_month(raw / "rsds_1hr_HOSTRADA-v1-0_BE_gn_2023010100-2023010123.nc", "2023-01-01", 24)
    _write_month(raw / "rsds_1hr_HOSTRADA-v1-0_BE_gn_2019010100-2019010123.nc", "2019-01-01", 24)
    return s


def test_find_nc_files_filters_years_and_sorts(settings: Settings) -> None:
    files = find_nc_files(settings.paths.weather_raw_dir, "rsds", [2023])
    assert [f.name[-24:-3] for f in files] == ["2023010100-2023010123", "2023020100-2023020123"]


def test_preprocess_creates_downsampled_time_ordered_cube(settings: Settings) -> None:
    preprocess_variable(settings, "rsds", chunk_hours=10)
    cube = WeatherCube.load(settings.paths.processed_dir, "rsds")

    assert cube.data.shape == (48, 4, 3)  # 8x6 grid, every 2nd cell
    assert cube.times.is_monotonic_increasing and cube.times.is_unique
    assert cube.times[0] == pd.Timestamp("2023-01-01") and cube.times[-1] == pd.Timestamp(
        "2023-02-01 23:00"
    )
    assert cube.data.dtype == np.float32
    assert np.isnan(cube.data[:, 0, 0]).all()
    assert cube.data[0, 1, 1] == 0.0 and cube.data[23, 1, 1] == 23.0  # January
    assert cube.data[24, 1, 1] == 100.0 and cube.data[47, 1, 1] == 123.0  # February


def test_preprocess_is_idempotent_unless_overwrite(settings: Settings) -> None:
    path = preprocess_variable(settings, "rsds")
    mtime = path.stat().st_mtime_ns
    preprocess_variable(settings, "rsds")
    assert path.stat().st_mtime_ns == mtime
    preprocess_variable(settings, "rsds", overwrite=True)
    assert path.stat().st_mtime_ns >= mtime


def test_missing_files_raise_helpful_error(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="renewcast download"):
        preprocess_variable(settings, "sfcWind")


def test_load_without_preprocessing_raises(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="renewcast preprocess"):
        WeatherCube.load(settings.paths.processed_dir, "rsds")


def test_select_files_matches_variable_and_year_exactly() -> None:
    names = [
        "sfcWind_1hr_HOSTRADA-v1-0_BE_gn_2023010100-2023013123.nc",
        "sfcWind_direction_1hr_HOSTRADA-v1-0_BE_gn_2023010100-2023013123.nc",
        "sfcWind_1hr_HOSTRADA-v1-0_BE_gn_2024110100-2024113023.nc",
        "sfcWind_1hr_HOSTRADA-v1-0_BE_gn_2022010100-2022013123.nc",
    ]
    picked = select_files(names, "sfcWind", "HOSTRADA-v1-0_BE_gn", [2023, 2024])
    assert picked == [names[0], names[2]]  # direction file and 2022 are excluded
