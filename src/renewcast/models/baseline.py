"""CPU-only reference models on spatial summary statistics.

Each weather frame is reduced to a handful of numbers (mean, spread, percentiles, maximum) plus
calendar features. A CNN that cannot beat gradient boosting on these features is not learning
anything useful from the spatial structure, so this is the sanity check for the GPU models.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from renewcast.config import Settings
from renewcast.data.prepare import prepare_target_data
from renewcast.data.weather import WeatherCube
from renewcast.evaluation.metrics import regression_report
from renewcast.utils import write_json

logger = logging.getLogger(__name__)

SUMMARY_NAMES = ["mean", "std", "p10", "p50", "p90", "max"]


def field_summary(cube: WeatherCube, cache_dir: Path, chunk: int = 64) -> np.ndarray:
    """Per-frame spatial statistics for every frame of the cube, cached on disk."""
    cache = cache_dir / "weather" / f"{cube.variable}_summary.npy"
    if cache.exists():
        cached = np.load(cache)
        if cached.shape == (len(cube.times), len(SUMMARY_NAMES)):
            return cached
    n = len(cube.times)
    out = np.empty((n, len(SUMMARY_NAMES)), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        for start in range(0, n, chunk):
            block = np.asarray(cube.data[start : start + chunk], dtype=np.float32)
            flat = block.reshape(len(block), -1)
            pct = np.nanpercentile(flat, [10, 50, 90], axis=1)
            out[start : start + len(block)] = np.column_stack(
                [
                    np.nanmean(flat, axis=1),
                    np.nanstd(flat, axis=1),
                    pct[0],
                    pct[1],
                    pct[2],
                    np.nanmax(flat, axis=1),
                ]
            )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def calendar_features(times: pd.DatetimeIndex) -> np.ndarray:
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60
    doy = times.dayofyear.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        ]
    ).astype(np.float32)


def run_baselines(settings: Settings, target_name: str) -> Path:
    """Fit Ridge and gradient boosting on summary features; write metrics and predictions."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    prepared = prepare_target_data(settings, target_name)
    summary = field_summary(prepared.cube, settings.paths.processed_dir)
    features = np.column_stack([summary[prepared.rows], calendar_features(prepared.times)])
    features = np.nan_to_num(features)
    y, splits = prepared.y, prepared.splits

    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=settings.runtime.seed
        ),
    }
    out_dir = settings.paths.baselines_dir / target_name
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    predictions = pd.DataFrame({"time_utc": prepared.times, "y_true": y})
    for name, model in models.items():
        model.fit(features[splits.train], y[splits.train])
        pred = model.predict(features)
        if settings.evaluation.clip_min_zero:
            pred = np.clip(pred, 0, None)
        predictions[name] = pred
        metrics[name] = {
            split: regression_report(y[idx], pred[idx])
            for split, idx in (("val", splits.val), ("test", splits.test))
        }
        logger.info("%s/%s test R2=%.4f", target_name, name, metrics[name]["test"]["r2"])

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    write_json(out_dir / "metrics.json", metrics)
    return out_dir
