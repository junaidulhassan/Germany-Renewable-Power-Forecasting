"""Command-line interface: ``renewcast <command> [--config configs/default.yaml]``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from renewcast import __version__, pipeline
from renewcast.config import Settings
from renewcast.utils import describe_environment, setup_logging

logger = logging.getLogger("renewcast")


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default="configs/default.yaml", help="YAML config file")
    common.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set targets.solar.training.epochs=3 (repeatable)",
    )
    common.add_argument(
        "-t",
        "--target",
        action="append",
        default=None,
        help="Target(s) to process (default: all in config), e.g. -t solar -t wind",
    )
    common.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING (default: config)")

    parser = argparse.ArgumentParser(
        prog="renewcast", description="Hourly solar & wind forecasting for Germany."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", parents=[common], help="Fetch HOSTRADA NetCDF files from DWD")
    p.add_argument("--dry-run", action="store_true", help="List files without downloading")

    sub.add_parser(
        "prepare-targets", parents=[common], help="Clean ENTSO-E CSV into hourly UTC targets"
    )

    p = sub.add_parser(
        "preprocess", parents=[common], help="Convert NetCDF into memmapped .npy cubes"
    )
    p.add_argument("--overwrite", action="store_true", help="Rebuild cubes that already exist")

    sub.add_parser("eda", parents=[common], help="Missing-value chart and weather maps")
    sub.add_parser(
        "baseline", parents=[common], help="CPU-only Ridge / gradient-boosting baselines"
    )

    p = sub.add_parser("train", parents=[common], help="Train CNN(s)")
    p.add_argument("--run-name", default=None, help="Run directory name (default: timestamp)")

    for name, text in (("evaluate", "Evaluate a trained run"), ("report", "Summarise results")):
        p = sub.add_parser(name, parents=[common], help=text)
        p.add_argument("--run", default=None, help="Run id or path (default: latest per target)")

    p = sub.add_parser("run-all", parents=[common], help="Run every stage in order")
    p.add_argument("--skip", nargs="*", default=[], choices=pipeline.STAGES, help="Stages to skip")
    p.add_argument("--overwrite", action="store_true")

    sub.add_parser("check-env", parents=[common], help="Print library versions and visible GPUs")
    sub.add_parser("show-config", parents=[common], help="Print the resolved configuration")
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")  # figures are written to disk, never shown
    args = _build_parser().parse_args(argv)
    settings = Settings.load(args.config, args.overrides)
    setup_logging(args.log_level or settings.runtime.log_level)

    targets = args.target or list(settings.targets)
    unknown = set(targets) - set(settings.targets)
    if unknown:
        logger.error(
            "Unknown target(s) %s; configured: %s", sorted(unknown), list(settings.targets)
        )
        return 2

    command = args.command
    if command == "download":
        pipeline.stage_download(settings, targets, dry_run=args.dry_run)
    elif command == "prepare-targets":
        pipeline.stage_prepare_targets(settings, targets)
    elif command == "preprocess":
        pipeline.stage_preprocess(settings, targets, overwrite=args.overwrite)
    elif command == "eda":
        pipeline.stage_eda(settings, targets)
    elif command == "baseline":
        pipeline.stage_baseline(settings, targets)
    elif command == "train":
        pipeline.stage_train(settings, targets, run_name=args.run_name)
    elif command == "evaluate":
        pipeline.stage_evaluate(settings, targets, run=args.run)
    elif command == "report":
        pipeline.stage_report(settings, targets, run=args.run)
    elif command == "run-all":
        pipeline.run_all(settings, targets, skip=args.skip, overwrite=args.overwrite)
    elif command == "check-env":
        print(json.dumps(describe_environment(), indent=2))
    elif command == "show-config":
        print(settings.to_yaml())
    return 0


if __name__ == "__main__":
    sys.exit(main())
