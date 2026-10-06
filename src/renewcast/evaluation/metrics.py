"""Regression metrics in physical units (MW)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R2, MAE, RMSE, bias and the MAE/RMSE normalised by mean observed generation."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    error = y_pred - y_true
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    mean_true = float(y_true.mean())
    scale = mean_true if mean_true > 0 else float("nan")
    return {
        "n": int(y_true.size),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mae": mae,
        "rmse": rmse,
        "bias": float(error.mean()),
        "nmae": mae / scale,
        "nrmse": rmse / scale,
        "mean_observed": mean_true,
    }


def monthly_report(times: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Metrics per calendar month (UTC), useful for spotting seasonal weaknesses."""
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=times)
    rows = []
    for period, group in frame.groupby(frame.index.to_period("M")):
        rows.append({"month": str(period), **regression_report(group.y_true, group.y_pred)})
    return pd.DataFrame(rows)
