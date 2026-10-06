"""Evaluate a trained run on its validation and test hours and write figures + metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from renewcast.config import Settings
from renewcast.data.prepare import prepare_target_data
from renewcast.data.scaling import InputScaler, TargetScaler
from renewcast.evaluation.metrics import monthly_report, regression_report
from renewcast.utils import read_json, resolve_run_dir, write_json
from renewcast.visualization import plots

logger = logging.getLogger(__name__)


def evaluate_target(settings: Settings, target_name: str, run: str | None = None) -> Path:
    """Score the model of ``run`` (default: latest) and write results into its directory."""
    import keras

    from renewcast.data.dataset import WeatherSequence

    run_dir = resolve_run_dir(settings.paths.runs_dir, target_name, run)
    logger.info("Evaluating %s", run_dir)

    # Splits, scaling and grid come from the run's own snapshot, not the current config.
    run_settings = Settings.from_yaml_text((run_dir / "config.yaml").read_text(encoding="utf-8"))
    run_settings.evaluation = settings.evaluation
    cfg = run_settings.targets[target_name]

    prepared = prepare_target_data(run_settings, target_name)
    input_scaler = InputScaler.load(run_dir / "input_scaler.npz")
    target_scaler = TargetScaler.load(run_dir / "target_scaler.npz")
    model = keras.saving.load_model(str(run_dir / "model.keras"))

    metrics: dict[str, dict[str, float]] = {}
    fig_dir = run_dir / "figures"
    eval_cfg = run_settings.evaluation
    name = cfg.display_name

    for split_name, idx in (("val", prepared.splits.val), ("test", prepared.splits.test)):
        sequence = WeatherSequence(
            prepared.cube.data,
            prepared.rows[idx],
            None,
            input_scaler,
            batch_size=eval_cfg.batch_size,
            shuffle=False,
        )
        scaled = model.predict(sequence, verbose=0).reshape(-1)
        y_pred = target_scaler.inverse(scaled)
        if eval_cfg.clip_min_zero:
            y_pred = np.clip(y_pred, 0.0, None)
        y_true = prepared.y[idx]
        times = prepared.times[idx]

        report = regression_report(y_true, y_pred)
        metrics[split_name] = report
        logger.info(
            "%s/%s: R2=%.4f MAE=%.1f %s RMSE=%.1f %s",
            target_name,
            split_name,
            report["r2"],
            report["mae"],
            cfg.unit,
            report["rmse"],
            cfg.unit,
        )

        local = times.tz_localize("UTC").tz_convert(eval_cfg.display_timezone)
        pd.DataFrame(
            {
                "time_utc": times,
                "time_local": local.tz_localize(None),
                "y_true": y_true,
                "y_pred": y_pred,
                "error": y_pred - y_true,
            }
        ).to_csv(run_dir / f"predictions_{split_name}.csv", index=False)

        tag = f"{target_name}_{split_name}"
        plots.plot_forecast_window(
            times,
            y_true,
            y_pred,
            eval_cfg.forecast_window_hours,
            f"{name}: observed vs predicted ({split_name}, first {eval_cfg.forecast_window_hours} h)",
            cfg.unit,
            fig_dir / f"{tag}_forecast_window.png",
        )
        plots.plot_daily_mean(
            times,
            y_true,
            y_pred,
            f"{name}: daily mean ({split_name})",
            cfg.unit,
            fig_dir / f"{tag}_daily_mean.png",
        )
        plots.plot_scatter(
            y_true,
            y_pred,
            report["r2"],
            f"{name} ({split_name})",
            cfg.unit,
            fig_dir / f"{tag}_scatter.png",
        )
        plots.plot_error_by_hour(
            local,
            y_pred - y_true,
            f"{name}: error by hour of day ({split_name})",
            cfg.unit,
            fig_dir / f"{tag}_error_by_hour.png",
        )
        if split_name == "test":
            monthly = monthly_report(times, y_true, y_pred)
            monthly.to_csv(run_dir / "metrics_monthly_test.csv", index=False)
            plots.plot_monthly_metric(
                monthly,
                "mae",
                f"{name}: MAE by month ({split_name})",
                fig_dir / f"{tag}_monthly_mae.png",
            )

    history_path = run_dir / "history.json"
    if history_path.exists():
        plots.plot_training_history(
            read_json(history_path), name, fig_dir / f"{target_name}_training_history.png"
        )

    write_json(run_dir / "metrics.json", metrics)
    return run_dir
