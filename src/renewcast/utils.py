"""Small shared helpers: logging, seeding, run directories, runtime configuration."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from renewcast.config import RuntimeConfig

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def configure_runtime(cfg: RuntimeConfig, mixed_precision: bool = False) -> None:
    """Configure GPU memory growth and numeric precision before any model is built."""
    try:
        import tensorflow as tf
    except ImportError:
        tf = None
    if tf is not None and cfg.gpu_memory_growth:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:  # raised if the GPU is already initialised
                logger.warning("Could not enable memory growth on %s: %s", gpu.name, exc)
    if mixed_precision:
        import keras

        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision enabled (mixed_float16)")


def describe_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for module in ("pandas", "xarray", "sklearn", "keras", "tensorflow"):
        try:
            info[module] = __import__(module).__version__
        except ImportError:
            info[module] = None
    try:
        import tensorflow as tf

        info["gpus"] = [g.name for g in tf.config.list_physical_devices("GPU")]
    except ImportError:
        info["gpus"] = []
    return info


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def create_run_dir(runs_dir: Path, target: str, name: str | None = None) -> Path:
    run_id = name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / target / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_run_dir(runs_dir: Path, target: str, run: str | None = None) -> Path:
    """Return the requested run directory, or the most recent one for ``target``."""
    base = runs_dir / target
    if run is not None:
        candidate = Path(run)
        if candidate.is_dir():
            return candidate
        candidate = base / run
        if not candidate.is_dir():
            raise FileNotFoundError(f"Run '{run}' not found under {base}")
        return candidate
    runs = sorted(p for p in base.glob("*") if p.is_dir()) if base.exists() else []
    if not runs:
        raise FileNotFoundError(
            f"No runs found for target '{target}' in {base}. Run `train` first."
        )
    return runs[-1]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
