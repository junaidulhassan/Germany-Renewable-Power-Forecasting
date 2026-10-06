"""Extract figures and reported metrics from an already-executed notebook.

The notebook is NEVER executed and NEVER modified: it is opened read-only and parsed as JSON.
Everything written here comes from outputs that were stored in the notebook when it was last run
(on a GPU machine), so no GPU or large RAM is needed.

Usage:
    python scripts/extract_notebook_assets.py notebooks/manisha-project_germany.ipynb
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

# Notebook cell index -> output file name (without extension).
FIGURES = {
    7: "eda_missing_values",
    20: "eda_weather_fields",
    45: "solar_training_history",
    46: "solar_validation_full",
    47: "solar_validation_zoom",
    60: "solar_forecast_2024",
    82: "wind_training_history",
    83: "wind_validation_full",
    84: "wind_validation_zoom",
    97: "wind_forecast_2024",
}
# Cells whose stdout holds the final numbers for each target.
METRIC_CELLS = {"solar": {"fit": 44, "r2": 46}, "wind": {"fit": 81, "r2": 83}}
SUMMARY_CELL = 43  # model.summary() of the (identical) architecture

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _stdout(cell: dict) -> str:
    return _ANSI.sub("", "".join("".join(o["text"]) for o in cell["outputs"] if "text" in o))


def _rich_text(cell: dict) -> str:
    parts = ["".join(o["data"]["text/plain"]) for o in cell["outputs"] if "data" in o]
    return _ANSI.sub("", "\n".join(parts))


def extract(notebook: Path, out_dir: Path, metrics_out: Path) -> None:
    with open(notebook, "rb") as fh:  # read-only
        cells = json.load(fh)["cells"]

    out_dir.mkdir(parents=True, exist_ok=True)
    for index, name in FIGURES.items():
        images = [
            o["data"]["image/png"]
            for o in cells[index]["outputs"]
            if "data" in o and "image/png" in o["data"]
        ]
        if not images:
            raise SystemExit(f"Cell {index} has no stored PNG output (expected '{name}')")
        (out_dir / f"{name}.png").write_bytes(base64.b64decode(images[0]))
        print(f"cell {index:3d} -> {out_dir / (name + '.png')}")

    params = re.search(r"Total params:\s*([\d,]+)", _rich_text(cells[SUMMARY_CELL]))
    results: dict[str, dict] = {}
    for target, spec in METRIC_CELLS.items():
        fit, r2 = _stdout(cells[spec["fit"]]), _stdout(cells[spec["r2"]])
        loss = re.search(r"Validation Loss: ([\d.]+), MAE: ([\d.]+)", fit)
        score = re.search(r"R\^2 Score: ([\d.]+)", r2)
        epochs = re.findall(r"Epoch (\d+)/(\d+)", fit)
        if not (loss and score):
            raise SystemExit(f"Could not find final metrics for '{target}' in the notebook")
        results[target] = {
            "validation_split": "random 10% of 2023 hours (shuffled)",
            "r2": float(score.group(1)),
            "val_loss_mse": float(loss.group(1)),
            "val_mae_mw": float(loss.group(2)),
            "epochs_logged": int(epochs[-1][0]) if epochs else None,
            "epochs_max": int(epochs[-1][1]) if epochs else None,
        }
    results["model_total_params"] = int(params.group(1).replace(",", "")) if params else None
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"metrics -> {metrics_out}\n{json.dumps(results, indent=2)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/images"))
    parser.add_argument("--metrics-out", type=Path, default=Path("docs/notebook_results.json"))
    args = parser.parse_args()
    extract(args.notebook, args.out_dir, args.metrics_out)


if __name__ == "__main__":
    main()
