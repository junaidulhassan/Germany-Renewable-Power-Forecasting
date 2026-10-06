"""End-to-end check of dataset, model, training and evaluation on a tiny synthetic problem.

Skipped automatically unless a Keras 3 backend (e.g. TensorFlow) is installed, so the CPU-only CI
job stays light. Run locally with ``pip install -e ".[tf,dev]" && pytest -m keras``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

keras = pytest.importorskip("keras")

from renewcast.config import ModelConfig, Settings  # noqa: E402
from renewcast.data.dataset import WeatherSequence  # noqa: E402
from renewcast.data.generation import save_targets  # noqa: E402
from renewcast.data.scaling import InputScaler  # noqa: E402
from renewcast.data.weather import cube_paths  # noqa: E402
from renewcast.evaluation.evaluate import evaluate_target  # noqa: E402
from renewcast.models.cnn import build_cnn  # noqa: E402
from renewcast.training.train import train_target  # noqa: E402
from tests.conftest import REPO_ROOT  # noqa: E402

pytestmark = pytest.mark.keras


def test_sequence_batches_and_keeps_order() -> None:
    data = np.random.default_rng(0).uniform(size=(20, 8, 6)).astype(np.float32)
    scaler = InputScaler("global_minmax").fit(data, np.arange(20))
    y = np.arange(20, dtype=np.float32)
    seq = WeatherSequence(data, np.arange(20), y, scaler, batch_size=8, shuffle=False)
    assert len(seq) == 3
    x0, y0 = seq[0]
    assert x0.shape == (8, 8, 6, 1) and y0.tolist() == list(range(8))
    assert seq[2][0].shape[0] == 4  # last partial batch

    shuffled = WeatherSequence(data, np.arange(20), y, scaler, batch_size=8, shuffle=True, seed=1)
    seen = np.concatenate([shuffled[i][1] for i in range(len(shuffled))])
    assert sorted(seen.tolist()) == list(range(20))  # every sample exactly once


@pytest.mark.parametrize("architecture", ["baseline", "compact"])
def test_models_output_one_value_per_frame(architecture: str) -> None:
    cfg = ModelConfig(architecture=architecture, filters=[4, 8])
    model = build_cnn((32, 24, 1), cfg)
    out = model.predict(np.zeros((3, 32, 24, 1), np.float32), verbose=0)
    assert out.shape == (3, 1)


def test_train_and_evaluate_on_synthetic_data(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    times = pd.date_range("2023-01-01", "2024-01-31 23:00", freq="h")
    n = len(times)
    brightness = rng.uniform(0, 1, size=n).astype(np.float32)
    frames = brightness[:, None, None] * np.ones((n, 16, 12), np.float32)
    frames += rng.normal(0, 0.01, size=frames.shape).astype(np.float32)

    settings = Settings.load(
        REPO_ROOT / "configs" / "default.yaml",
        [
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.interim_dir={tmp_path / 'interim'}",
            f"paths.runs_dir={tmp_path / 'runs'}",
            "split.train_years=[2023]",
            "split.test_years=[2024]",
            "targets.solar.model={architecture: compact, filters: [4, 8]}",
            "targets.solar.training={epochs: 3, patience: 2, batch_size: 64, workers: 1, "
            "max_train_samples: 1500}",
        ],
    )
    data_path, time_path, _ = cube_paths(settings.paths.processed_dir, "rsds")
    data_path.parent.mkdir(parents=True)
    np.save(data_path, frames)
    np.save(time_path, times.values.astype("datetime64[h]"))
    # generation is a simple monotone function of the frame brightness
    targets = pd.DataFrame({"solar": 1000 * brightness, "wind": 0.0}, index=times)
    save_targets(targets, settings.paths.interim_dir)

    run_dir = train_target(settings, "solar", run_name="synthetic")
    for name in (
        "model.keras",
        "config.yaml",
        "input_scaler.npz",
        "target_scaler.npz",
        "history.json",
        "splits.npz",
        "metadata.json",
    ):
        assert (run_dir / name).exists(), name

    evaluate_target(settings, "solar", run="synthetic")
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions_test.csv").exists()
    assert (run_dir / "figures" / "solar_test_scatter.png").exists()
    predictions = pd.read_csv(run_dir / "predictions_test.csv")
    assert len(predictions) == (times.year == 2024).sum()
    assert (predictions["y_pred"] >= 0).all()
