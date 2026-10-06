"""Evaluation figures. Each function saves a PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OBSERVED_COLOR = "black"
PREDICTED_COLOR = "red"


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_history(history: dict[str, list[float]], title: str, path: Path) -> Path:
    """Loss and MAE per epoch (values are in scaled target units)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, label in ((axes[0], "loss", "Loss"), (axes[1], "mae", "MAE")):
        ax.plot(history[key], label=f"Training {label}")
        ax.plot(history[f"val_{key}"], label=f"Validation {label}")
        ax.set_title(f"{title}: {label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
    return _save(fig, path)


def plot_forecast_window(
    times: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_hours: int,
    title: str,
    unit: str,
    path: Path,
) -> Path:
    """Observed vs predicted for the first ``n_hours`` of a split (as in the notebook)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times[:n_hours], y_pred[:n_hours], color=PREDICTED_COLOR, label="Predicted")
    ax.plot(times[:n_hours], y_true[:n_hours], color=OBSERVED_COLOR, label="Observed")
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(f"Generation ({unit})")
    ax.set_xlabel("Time")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, path)


def plot_daily_mean(
    times: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    unit: str,
    path: Path,
) -> Path:
    """Daily means over the whole split: shows seasonal bias that hourly plots hide."""
    frame = (
        pd.DataFrame({"observed": y_true, "predicted": y_pred}, index=times).resample("D").mean()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame.index, frame["predicted"], color=PREDICTED_COLOR, label="Predicted")
    ax.plot(frame.index, frame["observed"], color=OBSERVED_COLOR, label="Observed")
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(f"Daily mean generation ({unit})")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    return _save(fig, path)


def plot_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, r2: float, title: str, unit: str, path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.35)
    top = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([0, top], [0, top], color="gray", linestyle="--", label="Perfect prediction")
    ax.set_title(f"{title} (R² = {r2:.3f})", fontsize=13)
    ax.set_xlabel(f"Observed ({unit})")
    ax.set_ylabel(f"Predicted ({unit})")
    ax.legend()
    return _save(fig, path)


def plot_error_by_hour(
    times_local: pd.DatetimeIndex, error: np.ndarray, title: str, unit: str, path: Path
) -> Path:
    """Mean and spread of (predicted - observed) by local hour of day."""
    frame = pd.DataFrame({"error": error, "hour": times_local.hour})
    grouped = frame.groupby("hour")["error"]
    mean, std = grouped.mean(), grouped.std().fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(mean.index, mean.values, yerr=std.values, capsize=2, color="steelblue", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel(f"Mean error ± 1 std ({unit})")
    return _save(fig, path)


def plot_monthly_metric(monthly: pd.DataFrame, metric: str, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(monthly["month"], monthly[metric], color="slateblue", alpha=0.85)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, path)
