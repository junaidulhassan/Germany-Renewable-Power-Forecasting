"""Batch loader that streams scaled weather frames from the on-disk memmap.

Requires Keras 3 (``keras.utils.PyDataset`` is backend-agnostic).
"""

from __future__ import annotations

import keras
import numpy as np

from renewcast.data.scaling import InputScaler


class WeatherSequence(keras.utils.PyDataset):
    """Yields ``(x, y)`` batches with ``x`` shaped ``(batch, height, width, 1)``.

    ``rows`` indexes frames in ``data``; ``y`` is the (already scaled) target for each row, or
    ``None`` for inference. With ``shuffle=False`` the output order equals the input order, which
    ``Model.predict`` relies on.
    """

    def __init__(
        self,
        data: np.ndarray,
        rows: np.ndarray,
        y: np.ndarray | None,
        scaler: InputScaler,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if y is not None and len(y) != len(rows):
            raise ValueError("rows and y must have the same length")
        self._data = data
        self._rows = np.asarray(rows)
        self._y = None if y is None else np.asarray(y, dtype=np.float32)
        self._scaler = scaler
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self._order = np.arange(len(self._rows))
        if shuffle:
            self._rng.shuffle(self._order)

    def __len__(self) -> int:
        return int(np.ceil(len(self._rows) / self._batch_size))

    def __getitem__(self, index: int):
        start = index * self._batch_size
        # Sorting inside a batch keeps memmap reads sequential; sample order within a batch is
        # irrelevant for training, and is already ascending when shuffle=False.
        sel = np.sort(self._order[start : start + self._batch_size])
        x = self._scaler.transform(self._data[self._rows[sel]])[..., np.newaxis]
        if self._y is None:
            return x
        return x, self._y[sel]

    def on_epoch_end(self) -> None:
        if self._shuffle:
            self._rng.shuffle(self._order)
