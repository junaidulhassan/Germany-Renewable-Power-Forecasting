"""Train one CNN for one target and write a self-contained run directory.

Run directory contents (``artifacts/runs/<target>/<run_id>/``):

    config.yaml            exact settings used (evaluation reloads these)
    model.keras            weights of the best epoch
    input_scaler.npz       training-only weather scaling statistics
    target_scaler.npz      training-only target scaling statistics
    splits.npz             UTC hour of every train / val / test sample
    history.json           per-epoch loss and metrics
    training_log.csv       same, written live by Keras
    model_summary.txt      layer table
    metadata.json          versions, commit, sizes, timings
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from renewcast.config import Settings
from renewcast.data.prepare import PreparedData, prepare_target_data
from renewcast.data.scaling import InputScaler, TargetScaler
from renewcast.utils import (
    configure_runtime,
    create_run_dir,
    describe_environment,
    git_commit,
    set_seed,
    write_json,
)

logger = logging.getLogger(__name__)


def _subsample(train: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= len(train):
        return train
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(train, size=limit, replace=False))


def train_target(settings: Settings, target_name: str, run_name: str | None = None) -> Path:
    import keras

    from renewcast.data.dataset import WeatherSequence
    from renewcast.models.cnn import build_cnn, compile_model

    cfg = settings.targets[target_name]
    train_cfg = cfg.training
    configure_runtime(settings.runtime, train_cfg.mixed_precision)
    set_seed(settings.runtime.seed)

    prepared: PreparedData = prepare_target_data(settings, target_name)
    splits = prepared.splits
    train_idx = _subsample(splits.train, train_cfg.max_train_samples, settings.runtime.seed)

    logger.info("Fitting scalers on %d training hours", len(train_idx))
    input_scaler = InputScaler(cfg.scaling.input).fit(prepared.cube.data, prepared.rows[train_idx])
    target_scaler = TargetScaler(cfg.scaling.target).fit(prepared.y[train_idx])

    def make_sequence(idx: np.ndarray, shuffle: bool) -> WeatherSequence:
        return WeatherSequence(
            prepared.cube.data,
            prepared.rows[idx],
            target_scaler.transform(prepared.y[idx]),
            input_scaler,
            batch_size=train_cfg.batch_size,
            shuffle=shuffle,
            seed=settings.runtime.seed,
            workers=train_cfg.workers,
            max_queue_size=train_cfg.max_queue_size,
        )

    train_seq = make_sequence(train_idx, shuffle=True)
    val_seq = make_sequence(splits.val, shuffle=False)

    height, width = prepared.cube.frame_shape
    model = compile_model(build_cnn((height, width, 1), cfg.model), train_cfg)

    run_dir = create_run_dir(settings.paths.runs_dir, target_name, run_name)
    logger.info("Run directory: %s", run_dir)
    (run_dir / "config.yaml").write_text(settings.to_yaml(), encoding="utf-8")
    input_scaler.save(run_dir / "input_scaler.npz")
    target_scaler.save(run_dir / "target_scaler.npz")
    hours = {
        name: prepared.times[idx].values.astype("datetime64[h]").astype(np.int64)
        for name, idx in (("train", train_idx), ("val", splits.val), ("test", splits.test))
    }
    np.savez(run_dir / "splits.npz", **hours)

    summary_lines: list[str] = []
    model.summary(print_fn=lambda line, **_: summary_lines.append(line))
    (run_dir / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Model has %s parameters", f"{model.count_params():,}")

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=train_cfg.patience, restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
    ]
    if train_cfg.reduce_lr_patience:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=train_cfg.reduce_lr_patience
            )
        )
    if train_cfg.checkpoint:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(run_dir / "checkpoint.keras"), monitor="val_loss", save_best_only=True
            )
        )
    if train_cfg.tensorboard:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=str(run_dir / "tensorboard")))

    started = time.time()
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=train_cfg.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - started

    model.save(run_dir / "model.keras")
    write_json(run_dir / "history.json", history.history)
    val_loss = history.history["val_loss"]
    write_json(
        run_dir / "metadata.json",
        {
            "target": target_name,
            "commit": git_commit(),
            "environment": describe_environment(),
            "n_parameters": int(model.count_params()),
            "sizes": {"train": len(train_idx), "val": len(splits.val), "test": len(splits.test)},
            "epochs_run": len(val_loss),
            "best_epoch": int(np.argmin(val_loss)) + 1,
            "best_val_loss_scaled": float(np.min(val_loss)),
            "train_seconds": round(elapsed, 1),
        },
    )
    logger.info(
        "Finished %s in %.0fs, best epoch %d", target_name, elapsed, int(np.argmin(val_loss)) + 1
    )
    return run_dir
