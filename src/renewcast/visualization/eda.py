"""Exploratory figures reproduced from the original notebook."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from renewcast.config import Settings
from renewcast.data.generation import count_missing
from renewcast.data.weather import find_nc_files

logger = logging.getLogger(__name__)

_COLORMAPS = {"rsds": "viridis", "sfcWind": "coolwarm", "sfcWind_direction": "twilight"}


def plot_missing_values(settings: Settings, path: Path) -> Path:
    counts = count_missing(settings)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [("#d62728", "black")[i % 2] for i in range(len(counts))]
    bars = ax.bar(counts["Column"], counts["Null Count"], color=colors)
    for bar, value in zip(bars, counts["Null Count"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value)}", ha="center", va="bottom")
    ax.set_title("Missing Values per Column", fontsize=14)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Number of Missing Values")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_weather_maps(settings: Settings, path: Path, variables: list[str] | None = None) -> Path:
    """One map per variable for a single hour of the first available monthly file."""
    import xarray as xr

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        ccrs = cfeature = None
        logger.warning(
            "cartopy not installed; drawing maps without coastlines (pip install '.[viz]')"
        )

    variables = variables or settings.download.variables
    index = settings.eda.map_time_index
    fig, axes = plt.subplots(
        len(variables),
        1,
        figsize=(10, 6 * len(variables)),
        subplot_kw={"projection": ccrs.PlateCarree()} if ccrs else {},
    )
    axes = [axes] if len(variables) == 1 else list(axes)
    for ax, variable in zip(axes, variables, strict=True):
        spec = settings.weather.variables[variable]
        files = find_nc_files(settings.paths.weather_raw_dir, variable, settings.years)
        if not files:
            raise FileNotFoundError(f"No NetCDF files for '{variable}'; run `download` first")
        with xr.open_dataset(files[0]) as ds:
            field = ds[variable].isel(time=index)
            lon, lat = ds["lon"].values, ds["lat"].values
            values = field.values
            stamp = str(ds["time"].values[index])[:16]
        kwargs = {"transform": ccrs.PlateCarree()} if ccrs else {}
        mesh = ax.contourf(lon, lat, values, 60, cmap=_COLORMAPS.get(variable, "viridis"), **kwargs)
        if ccrs:
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        fig.colorbar(mesh, ax=ax, orientation="horizontal", label=f"{spec.label} ({spec.units})")
        ax.set_title(f"{spec.label} - {stamp} UTC")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
