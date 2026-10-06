"""Score the 2024 predictions saved by the original notebook against the observed generation.

The notebook only reported R2 on a shuffled hold-out of 2023; it never scored its own 2024
forecasts. This script does, under two timestamp conventions:

* ``notebook``: prediction hour i is compared with the generation value whose *local-time*
  label equals the prediction's label (what the notebook implicitly did).
* ``utc``: the weather frames are in UTC, so the observed generation is first converted from
  Europe/Berlin local time to UTC.

Needs only the CPU-light ``renewcast`` package (pandas, numpy). Reads files, writes nothing
except an optional markdown table.

Usage:
    python scripts/rescore_saved_predictions.py \
        --generation 20200101-20241030-gen_per_prod_type.csv \
        --solar solar_power.csv --wind wind_power.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from renewcast.config import Settings
from renewcast.data.generation import build_targets
from renewcast.evaluation.metrics import regression_report


def _local_label_truth(generation: Path) -> pd.DataFrame:
    """Hourly observed values keyed by their local-time label, as in the notebook."""
    raw = pd.read_csv(generation, usecols=["MTU", "solar", "wind_onshore", "wind_offshore"])
    raw["MTU"] = pd.to_datetime(raw["MTU"])
    raw = raw[raw["MTU"].dt.year.isin([2023, 2024]) & (raw["MTU"].dt.minute == 0)]
    raw = raw.drop_duplicates("MTU").set_index("MTU").interpolate(limit_direction="both")
    return pd.DataFrame({"solar": raw["solar"], "wind": raw["wind_onshore"] + raw["wind_offshore"]})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--solar", type=Path, required=True)
    parser.add_argument("--wind", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings.load(
        args.config,
        [
            f"paths.raw_generation={args.generation}",
            "generation.hourly_aggregation=instant",
            "generation.imputation=interpolate",
        ],
    )
    truths = {
        "notebook (local-time labels)": _local_label_truth(args.generation),
        "UTC-aligned": build_targets(settings),
    }
    rows = []
    for target, path, column in (("solar", args.solar, "Solar"), ("wind", args.wind, "Wind_Power")):
        prediction = pd.read_csv(path, parse_dates=["MTU"]).set_index("MTU")[column]
        for convention, truth in truths.items():
            joined = pd.concat([truth[target], prediction], axis=1, join="inner").dropna()
            report = regression_report(joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy())
            rows.append({"target": target, "alignment": convention, **report})

    table = pd.DataFrame(rows)[["target", "alignment", "n", "r2", "mae", "rmse", "nmae"]]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.markdown_out:
        lines = [
            "| Target | Alignment | Hours | R² | MAE (MW) | RMSE (MW) | nMAE |",
            "|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {r.target} | {r.alignment} | {r.n} | {r.r2:.3f} | {r.mae:,.0f} | {r.rmse:,.0f} | {r.nmae:.3f} |"
            for r in table.itertuples()
        ]
        args.markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
