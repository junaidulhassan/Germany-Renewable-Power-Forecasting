"""Convert monthly HOSTRADA NetCDF files into one memory-mapped ``.npy`` cube per variable.

The notebook held every array in RAM (32-64 GB). Here each month is streamed in small time chunks
straight into a pre-allocated memmap on disk, so peak memory is a few hundred MB regardless of
how many years are processed. Training later reads batches from the memmap on demand.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from renewcast.config import Settings
from renewcast.utils import write_json

logger = logging.getLogger(__name__)

_MONTH_RE = re.compile(r"_(\d{4})(\d{2})0100-")


def cube_paths(processed_dir: Path, variable: str) -> tuple[Path, Path, Path]:
    base = processed_dir / "weather"
    return base / f"{variable}.npy", base / f"{variable}_time.npy", base / f"{variable}_meta.json"


def find_nc_files(raw_dir: Path, variable: str, years: list[int]) -> list[Path]:
    directory = raw_dir / variable
    files = []
    for path in directory.glob("*.nc"):
        match = _MONTH_RE.search(path.name)
        if match and int(match.group(1)) in years:
            files.append(path)
    return sorted(files)


def _to_hours(times: np.ndarray, source: Path) -> np.ndarray:
    ns = times.astype("datetime64[ns]")
    hours = ns.astype("datetime64[h]")
    if not np.array_equal(ns, hours.astype("datetime64[ns]")):
        logger.warning("%s: timestamps are not on the hour; flooring to the hour", source.name)
    return hours


def preprocess_variable(
    settings: Settings,
    variable: str,
    overwrite: bool = False,
    chunk_hours: int = 48,
) -> Path:
    """Stack all monthly files of ``variable`` into ``processed/weather/<variable>.npy``."""
    import xarray as xr

    data_path, time_path, meta_path = cube_paths(settings.paths.processed_dir, variable)
    if data_path.exists() and time_path.exists() and not overwrite:
        logger.info(
            "%s: cube already exists at %s (use --overwrite to rebuild)", variable, data_path
        )
        return data_path

    files = find_nc_files(settings.paths.weather_raw_dir, variable, settings.years)
    if not files:
        raise FileNotFoundError(
            f"No NetCDF files for '{variable}' under {settings.paths.weather_raw_dir / variable} "
            f"for years {settings.years}. Run `renewcast download` first."
        )
    factor = settings.weather.downsample_factor

    # Pass 1: read only metadata to learn the final shape and the time axis.
    time_chunks: list[np.ndarray] = []
    spatial_shape: tuple[int, int] | None = None
    for path in files:
        with xr.open_dataset(path) as ds:
            if variable not in ds:
                raise KeyError(f"Variable '{variable}' not found in {path.name}")
            da = ds[variable]
            if da.ndim != 3 or da.dims[0] != "time":
                raise ValueError(f"{path.name}: expected dims (time, y, x), got {da.dims}")
            shape = (int(da.shape[1]), int(da.shape[2]))
            if spatial_shape is None:
                spatial_shape = shape
            elif shape != spatial_shape:
                raise ValueError(f"{path.name}: grid {shape} differs from {spatial_shape}")
            time_chunks.append(_to_hours(ds["time"].values, path))
    assert spatial_shape is not None

    times = np.concatenate(time_chunks)
    if not np.all(np.diff(times.astype(np.int64)) > 0):
        raise ValueError(f"{variable}: time axis is not strictly increasing across files")
    height = len(range(0, spatial_shape[0], factor))
    width = len(range(0, spatial_shape[1], factor))
    logger.info(
        "%s: %d files, %d hours, grid %s -> (%d, %d)",
        variable,
        len(files),
        len(times),
        spatial_shape,
        height,
        width,
    )

    # Pass 2: stream chunks into the memmap.
    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = data_path.with_name(data_path.name + ".tmp")
    cube = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.float32, shape=(len(times), height, width)
    )
    cursor = 0
    for path in files:
        with xr.open_dataset(path) as ds:
            da = ds[variable]
            n_time = int(da.sizes["time"])
            for start in range(0, n_time, chunk_hours):
                block = da.isel(time=slice(start, start + chunk_hours)).values
                block = block[:, ::factor, ::factor]  # masked cells are already NaN
                cube[cursor + start : cursor + start + block.shape[0]] = block
            cursor += n_time
        logger.info("%s: wrote %s (%d/%d hours)", variable, path.name, cursor, len(times))
    cube.flush()
    del cube
    os.replace(tmp_path, data_path)

    np.save(time_path, times)
    write_json(
        meta_path,
        {
            "variable": variable,
            "shape": [len(times), height, width],
            "downsample_factor": factor,
            "source_grid": list(spatial_shape),
            "first_time": str(times[0]),
            "last_time": str(times[-1]),
            "files": [p.name for p in files],
        },
    )
    return data_path


@dataclass(frozen=True)
class WeatherCube:
    """Read-only handle on a preprocessed variable."""

    variable: str
    data: np.ndarray  # memmap, shape (time, height, width)
    times: pd.DatetimeIndex  # UTC, unique, increasing

    @classmethod
    def load(cls, processed_dir: Path, variable: str) -> WeatherCube:
        data_path, time_path, _ = cube_paths(processed_dir, variable)
        if not data_path.exists():
            raise FileNotFoundError(f"{data_path} not found. Run `renewcast preprocess` first.")
        data = np.load(data_path, mmap_mode="r")
        times = pd.DatetimeIndex(np.load(time_path).astype("datetime64[ns]"))
        if len(times) != data.shape[0]:
            raise ValueError(f"{variable}: {len(times)} timestamps but {data.shape[0]} frames")
        if not times.is_unique:
            raise ValueError(f"{variable}: duplicated timestamps in cube")
        return cls(variable=variable, data=data, times=times)

    @property
    def frame_shape(self) -> tuple[int, int]:
        return int(self.data.shape[1]), int(self.data.shape[2])
