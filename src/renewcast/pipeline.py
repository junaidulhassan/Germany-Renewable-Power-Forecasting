"""Stage orchestration shared by the CLI sub-commands and ``run-all``."""

from __future__ import annotations

import logging
from pathlib import Path

from renewcast.config import Settings

logger = logging.getLogger(__name__)

# Order matters: each stage consumes the outputs of the ones before it.
STAGES = [
    "download",
    "prepare-targets",
    "preprocess",
    "eda",
    "baseline",
    "train",
    "evaluate",
    "report",
]


def stage_download(settings: Settings, targets: list[str], dry_run: bool = False) -> None:
    from renewcast.data.download import download_weather

    download_weather(settings, settings.weather_variables_for(targets), dry_run=dry_run)


def stage_prepare_targets(settings: Settings, targets: list[str]) -> Path:
    from renewcast.data.generation import build_targets, save_targets

    frame = build_targets(settings)
    path = save_targets(frame, settings.paths.interim_dir)
    logger.info("Saved hourly targets to %s", path)
    return path


def stage_preprocess(settings: Settings, targets: list[str], overwrite: bool = False) -> None:
    from renewcast.data.weather import preprocess_variable

    for variable in settings.weather_variables_for(targets):
        preprocess_variable(settings, variable, overwrite=overwrite)


def stage_eda(settings: Settings, targets: list[str]) -> None:
    from renewcast.visualization.eda import plot_missing_values, plot_weather_maps

    out = settings.paths.reports_dir / "figures" / "eda"
    plot_missing_values(settings, out / "missing_values.png")
    plot_weather_maps(settings, out / "weather_fields.png")


def stage_baseline(settings: Settings, targets: list[str]) -> None:
    from renewcast.models.baseline import run_baselines

    for target in targets:
        run_baselines(settings, target)


def stage_train(
    settings: Settings, targets: list[str], run_name: str | None = None
) -> dict[str, Path]:
    from renewcast.training.train import train_target

    return {t: train_target(settings, t, run_name) for t in targets}


def stage_evaluate(settings: Settings, targets: list[str], run: str | None = None) -> None:
    from renewcast.evaluation.evaluate import evaluate_target

    for target in targets:
        evaluate_target(settings, target, run)


def stage_report(settings: Settings, targets: list[str], run: str | None = None) -> None:
    from renewcast.evaluation.report import build_report

    build_report(settings, targets, run)


def run_all(
    settings: Settings,
    targets: list[str],
    skip: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Run every stage in order, optionally skipping some (e.g. ``--skip download eda``)."""
    skip = set(skip or [])
    unknown = skip - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stages to skip: {sorted(unknown)}; valid: {STAGES}")
    for stage in STAGES:
        if stage in skip:
            logger.info("=== Skipping stage: %s ===", stage)
            continue
        logger.info("=== Stage: %s ===", stage)
        if stage == "download":
            stage_download(settings, targets)
        elif stage == "prepare-targets":
            stage_prepare_targets(settings, targets)
        elif stage == "preprocess":
            stage_preprocess(settings, targets, overwrite)
        elif stage == "eda":
            stage_eda(settings, targets)
        elif stage == "baseline":
            stage_baseline(settings, targets)
        elif stage == "train":
            stage_train(settings, targets)
        elif stage == "evaluate":
            stage_evaluate(settings, targets)
        elif stage == "report":
            stage_report(settings, targets)
