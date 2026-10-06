"""Collect the latest results of every target into ``reports/``."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from renewcast.config import Settings
from renewcast.utils import read_json, resolve_run_dir

logger = logging.getLogger(__name__)


def build_report(settings: Settings, targets: list[str], run: str | None = None) -> Path:
    """Write ``reports/metrics_summary.md`` and copy each target's figures to ``reports/figures``."""
    reports = settings.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Results summary",
        "",
        "| Target | Run | Split | R² | MAE (MW) | RMSE (MW) | nMAE | Bias (MW) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for target in targets:
        run_dir = resolve_run_dir(settings.paths.runs_dir, target, run)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            logger.warning("%s has no metrics.json; run `evaluate` first", run_dir)
            continue
        for split, m in read_json(metrics_path).items():
            lines.append(
                f"| {settings.targets[target].display_name} | {run_dir.name} | {split} | "
                f"{m['r2']:.4f} | {m['mae']:.0f} | {m['rmse']:.0f} | {m['nmae']:.3f} | {m['bias']:.0f} |"
            )
        figures = run_dir / "figures"
        if figures.exists():
            dest = reports / "figures" / target
            dest.mkdir(parents=True, exist_ok=True)
            for png in figures.glob("*.png"):
                shutil.copy2(png, dest / png.name)
    summary = reports / "metrics_summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", summary)
    return summary
