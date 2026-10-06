from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from renewcast.data.scaling import InputScaler, TargetScaler


@pytest.fixture
def cube() -> np.ndarray:
    rng = np.random.default_rng(0)
    data = rng.uniform(10, 50, size=(40, 6, 5)).astype(np.float32)
    data[:, 0, 0] = np.nan  # a cell outside Germany
    data[:, 1, 1] = 7.0  # a constant cell
    return data


def test_pixel_minmax_maps_training_data_to_unit_interval(cube: np.ndarray) -> None:
    rows = np.arange(40)
    scaler = InputScaler("pixel_minmax").fit(cube, rows)
    out = scaler.transform(cube)
    assert out.shape == cube.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isfinite(out).all()
    assert (out[:, 0, 0] == 0.0).all()  # NaN cell -> 0
    assert (out[:, 1, 1] == 0.0).all()  # constant cell does not divide by zero
    assert out[:, 3, 3].max() == pytest.approx(1.0)


def test_scaler_uses_only_the_rows_it_was_fitted_on(cube: np.ndarray) -> None:
    cube = cube.copy()
    cube[39] = 1000.0  # outlier outside the training rows
    scaler = InputScaler("global_minmax").fit(cube, np.arange(30))
    assert float(scaler.scale) < 100  # the outlier did not leak into the statistics
    assert scaler.transform(cube[39:40]).max() > 1.0  # unseen data may exceed [0, 1]


def test_global_standard_has_zero_mean_unit_std(cube: np.ndarray) -> None:
    scaler = InputScaler("global_standard").fit(cube, np.arange(40))
    out = scaler.transform(cube)
    valid = ~np.isnan(cube)
    assert out[valid].mean() == pytest.approx(0.0, abs=1e-3)
    assert out[valid].std() == pytest.approx(1.0, abs=1e-2)


def test_input_scaler_save_load_roundtrip(cube: np.ndarray, tmp_path: Path) -> None:
    for mode in ("pixel_minmax", "global_minmax", "global_standard"):
        scaler = InputScaler(mode).fit(cube, np.arange(40))
        path = tmp_path / f"{mode}.npz"
        scaler.save(path)
        loaded = InputScaler.load(path)
        assert loaded.mode == mode
        assert np.allclose(loaded.transform(cube), scaler.transform(cube))


def test_unfitted_scaler_raises(cube: np.ndarray) -> None:
    with pytest.raises(RuntimeError):
        InputScaler("pixel_minmax").transform(cube)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        InputScaler("nope")
    with pytest.raises(ValueError):
        TargetScaler("nope")


def test_target_scaler_roundtrip(tmp_path: Path) -> None:
    y = np.array([0.0, 5000.0, 10000.0, 20000.0])
    scaler = TargetScaler("standard").fit(y)
    z = scaler.transform(y)
    assert z.mean() == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(scaler.inverse(z), y, atol=1e-2)
    scaler.save(tmp_path / "t.npz")
    assert np.allclose(TargetScaler.load(tmp_path / "t.npz").inverse(z), y, atol=1e-2)


def test_target_scaler_none_is_identity() -> None:
    y = np.array([1.0, 2.0, 3.0])
    scaler = TargetScaler("none").fit(y)
    assert np.allclose(scaler.transform(y), y)
    assert np.allclose(scaler.inverse(y), y)
