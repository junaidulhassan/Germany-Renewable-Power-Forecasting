"""Typed, validated configuration.

All behaviour of the pipeline is driven by one YAML file (see ``configs/default.yaml``).
Unknown keys are rejected so typos fail loudly instead of being silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    raw_generation: Path = Path("data/raw/generation/20200101-20241030-gen_per_prod_type.csv")
    weather_raw_dir: Path = Path("data/raw/weather")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    runs_dir: Path = Path("artifacts/runs")
    baselines_dir: Path = Path("artifacts/baselines")
    reports_dir: Path = Path("reports")


class WeatherVariable(_Strict):
    """One HOSTRADA variable. The dict key in ``WeatherConfig.variables`` is also the NetCDF name."""

    directory: str
    label: str
    units: str


class WeatherConfig(_Strict):
    base_url: str = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada"
    file_stem: str = "HOSTRADA-v1-0_BE_gn"
    # Keep every n-th grid point in both spatial directions (938x720 -> 469x360 for n=2).
    downsample_factor: int = Field(2, ge=1)
    variables: dict[str, WeatherVariable]


class DownloadConfig(_Strict):
    variables: list[str] = ["rsds", "sfcWind", "sfcWind_direction"]
    timeout_s: int = 120
    max_retries: int = 3
    workers: int = Field(4, ge=1)
    skip_existing: bool = True


class GenerationConfig(_Strict):
    timestamp_column: str = "MTU"
    # ENTSO-E market time units are local time (CET/CEST); HOSTRADA grids are UTC.
    source_timezone: str = "Europe/Berlin"
    # "instant": keep the value stamped on the hour (notebook behaviour).
    # "mean": average the four quarter-hour values of each hour.
    hourly_aggregation: Literal["instant", "mean"] = "mean"
    imputation: Literal["interpolate", "iterative", "none"] = "interpolate"


class SplitConfig(_Strict):
    train_years: list[int] = [2023]
    test_years: list[int] = [2024]
    # random: shuffled hours (notebook behaviour, leaks neighbouring hours into validation)
    # tail:   last `val_fraction` of the training period
    # blocked: random contiguous blocks of `block_hours`
    val_strategy: Literal["blocked", "tail", "random"] = "blocked"
    val_fraction: float = Field(0.1, gt=0.0, lt=1.0)
    block_hours: int = Field(168, ge=1)
    # Training hours closer than this to any validation hour are dropped (not for "random").
    gap_hours: int = Field(24, ge=0)
    seed: int = 42

    @model_validator(mode="after")
    def _disjoint_years(self) -> SplitConfig:
        overlap = set(self.train_years) & set(self.test_years)
        if overlap:
            raise ValueError(f"train_years and test_years overlap: {sorted(overlap)}")
        return self


class ModelConfig(_Strict):
    # baseline: the notebook CNN (2x Conv+Pool -> Flatten -> Dense 128 -> Dropout -> Dense 1)
    # compact:  deeper conv stack with BatchNorm, far fewer parameters
    architecture: Literal["baseline", "compact"] = "baseline"
    filters: list[int] = [32, 64]
    dense_units: int = 128
    dropout: float = Field(0.5, ge=0.0, lt=1.0)


class ScalingConfig(_Strict):
    input: Literal["pixel_minmax", "global_minmax", "global_standard"] = "pixel_minmax"
    target: Literal["standard", "none"] = "standard"


class TrainingConfig(_Strict):
    epochs: int = Field(50, ge=1)
    batch_size: int = Field(32, ge=1)
    learning_rate: float = Field(1e-3, gt=0)
    loss: Literal["mse", "mae", "huber"] = "mse"
    patience: int = Field(5, ge=1)
    reduce_lr_patience: int | None = None
    mixed_precision: bool = False
    checkpoint: bool = True
    tensorboard: bool = False
    workers: int = Field(2, ge=1)
    max_queue_size: int = Field(8, ge=1)
    # Debug aid: train on a random subset of this many hours.
    max_train_samples: int | None = Field(None, ge=1)


class TargetConfig(_Strict):
    display_name: str
    weather_variable: str
    generation_columns: list[str]
    unit: str = "MW"
    model: ModelConfig = ModelConfig()
    scaling: ScalingConfig = ScalingConfig()
    training: TrainingConfig = TrainingConfig()


class EvaluationConfig(_Strict):
    batch_size: int = Field(64, ge=1)
    # Generation cannot be negative, so predictions are floored at zero.
    clip_min_zero: bool = True
    forecast_window_hours: int = Field(300, ge=24)
    display_timezone: str = "Europe/Berlin"


class EdaConfig(_Strict):
    map_time_index: int = 32
    missing_values_year_filter: bool = True


class RuntimeConfig(_Strict):
    seed: int = 42
    log_level: str = "INFO"
    gpu_memory_growth: bool = True


class Settings(_Strict):
    paths: PathsConfig = PathsConfig()
    weather: WeatherConfig
    download: DownloadConfig = DownloadConfig()
    generation: GenerationConfig = GenerationConfig()
    split: SplitConfig = SplitConfig()
    targets: dict[str, TargetConfig]
    evaluation: EvaluationConfig = EvaluationConfig()
    eda: EdaConfig = EdaConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    @model_validator(mode="after")
    def _cross_checks(self) -> Settings:
        known = set(self.weather.variables)
        for name, target in self.targets.items():
            if target.weather_variable not in known:
                raise ValueError(
                    f"target '{name}' uses unknown weather variable "
                    f"'{target.weather_variable}' (known: {sorted(known)})"
                )
        unknown = set(self.download.variables) - known
        if unknown:
            raise ValueError(f"download.variables has unknown entries: {sorted(unknown)}")
        return self

    @property
    def years(self) -> list[int]:
        return sorted(set(self.split.train_years) | set(self.split.test_years))

    def weather_variables_for(self, targets: list[str]) -> list[str]:
        return sorted({self.targets[t].weather_variable for t in targets})

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> Settings:
        raw = load_yaml_with_extends(Path(path))
        apply_overrides(raw, overrides or [])
        return cls.model_validate(raw)

    @classmethod
    def from_yaml_text(cls, text: str) -> Settings:
        return cls.model_validate(yaml.safe_load(text) or {})


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into a copy of ``base``; non-dict values are replaced."""
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_with_extends(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML file; an optional top-level ``extends: other.yaml`` is resolved relative to it."""
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular 'extends' chain: {' -> '.join(map(str, (*_seen, path)))}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return deep_merge(load_yaml_with_extends(path.parent / parent, (*_seen, path)), raw)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides in place. Values are parsed as YAML scalars/lists."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        dotted, _, text = item.partition("=")
        keys = dotted.strip().split(".")
        node = raw
        for key in keys[:-1]:
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override '{dotted}': '{key}' is not a section")
            node = child
        node[keys[-1]] = yaml.safe_load(text)
    return raw
