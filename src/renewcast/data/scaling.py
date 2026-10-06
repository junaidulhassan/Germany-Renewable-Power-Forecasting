"""Scalers fitted on training data only and persisted with each run.

The notebook refitted a ``MinMaxScaler`` on the 2024 data before predicting, so training and
inference used different scalings. Here statistics come from the training hours alone and are
stored in the run directory, so evaluation applies exactly the same transform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_EPS = 1e-6


class InputScaler:
    """Scales weather frames as ``(x - lo) / scale`` and maps NaN (outside Germany) to 0."""

    def __init__(self, mode: str) -> None:
        if mode not in {"pixel_minmax", "global_minmax", "global_standard"}:
            raise ValueError(f"Unknown input scaling '{mode}'")
        self.mode = mode
        self.lo: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, data: np.ndarray, rows: np.ndarray, chunk: int = 64) -> InputScaler:
        rows = np.sort(np.asarray(rows))
        if self.mode == "global_standard":
            total = total_sq = 0.0
            count = 0
            for i in range(0, len(rows), chunk):
                block = np.asarray(data[rows[i : i + chunk]], dtype=np.float64)
                valid = ~np.isnan(block)
                count += int(valid.sum())
                total += float(block[valid].sum())
                total_sq += float(np.square(block[valid]).sum())
            mean = total / max(count, 1)
            std = float(np.sqrt(max(total_sq / max(count, 1) - mean**2, 0.0)))
            self.lo = np.float32(mean)
            self.scale = np.float32(std if std > _EPS else 1.0)
            return self

        axis = 0 if self.mode == "pixel_minmax" else None
        lo = hi = None
        for i in range(0, len(rows), chunk):
            block = np.asarray(data[rows[i : i + chunk]], dtype=np.float32)
            block_lo = np.fmin.reduce(block, axis=axis)  # fmin/fmax skip NaN
            block_hi = np.fmax.reduce(block, axis=axis)
            lo = block_lo if lo is None else np.fmin(lo, block_lo)
            hi = block_hi if hi is None else np.fmax(hi, block_hi)
        assert lo is not None and hi is not None
        lo = np.nan_to_num(lo, nan=0.0)  # pixels that are always NaN
        hi = np.where(np.isnan(hi), lo, hi)
        span = hi - lo
        self.lo = np.asarray(lo, dtype=np.float32)
        self.scale = np.asarray(np.where(span > _EPS, span, 1.0), dtype=np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.lo is None or self.scale is None:
            raise RuntimeError("InputScaler is not fitted")
        out = (np.asarray(x, dtype=np.float32) - self.lo) / self.scale
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def save(self, path: Path) -> None:
        if self.lo is None or self.scale is None:
            raise RuntimeError("InputScaler is not fitted")
        np.savez(path, mode=np.array(self.mode), lo=self.lo, scale=self.scale)

    @classmethod
    def load(cls, path: Path) -> InputScaler:
        with np.load(path, allow_pickle=False) as z:
            scaler = cls(str(z["mode"]))
            scaler.lo, scaler.scale = z["lo"], z["scale"]
        return scaler


class TargetScaler:
    """Standardises the generation target (MW) so the loss is O(1) instead of O(1e7)."""

    def __init__(self, mode: str) -> None:
        if mode not in {"standard", "none"}:
            raise ValueError(f"Unknown target scaling '{mode}'")
        self.mode = mode
        self.mean = 0.0
        self.std = 1.0

    def fit(self, y: np.ndarray) -> TargetScaler:
        if self.mode == "standard":
            self.mean = float(np.mean(y))
            std = float(np.std(y))
            self.std = std if std > _EPS else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((np.asarray(y, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64) * self.std + self.mean

    def save(self, path: Path) -> None:
        np.savez(path, mode=np.array(self.mode), mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> TargetScaler:
        with np.load(path, allow_pickle=False) as z:
            scaler = cls(str(z["mode"]))
            scaler.mean, scaler.std = float(z["mean"]), float(z["std"])
        return scaler
