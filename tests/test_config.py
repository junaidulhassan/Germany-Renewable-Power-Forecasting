from __future__ import annotations

import pytest
from pydantic import ValidationError

from renewcast.config import Settings, apply_overrides, deep_merge
from tests.conftest import REPO_ROOT


def test_default_config_loads(default_settings: Settings) -> None:
    assert set(default_settings.targets) == {"solar", "wind"}
    assert default_settings.years == [2023, 2024]
    assert default_settings.targets["wind"].generation_columns == ["wind_onshore", "wind_offshore"]


@pytest.mark.parametrize("name", ["notebook_replica.yaml", "smoke_test.yaml"])
def test_derived_configs_inherit_and_override(name: str) -> None:
    settings = Settings.load(REPO_ROOT / "configs" / name)
    assert settings.weather.variables["rsds"].directory == "radiation_downwelling"  # inherited
    assert set(settings.targets) == {"solar", "wind"}


def test_notebook_replica_values() -> None:
    settings = Settings.load(REPO_ROOT / "configs" / "notebook_replica.yaml")
    assert settings.split.val_strategy == "random"
    assert settings.generation.hourly_aggregation == "instant"
    assert settings.targets["solar"].scaling.target == "none"
    # untouched values still come from default.yaml
    assert settings.targets["solar"].training.epochs == 15


def test_unknown_key_is_rejected() -> None:
    bad = (REPO_ROOT / "configs" / "default.yaml").read_text().replace("split:", "splitt:", 1)
    with pytest.raises(ValidationError):
        Settings.from_yaml_text(bad)


def test_target_with_unknown_weather_variable_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown weather variable"):
        Settings.load(
            REPO_ROOT / "configs" / "default.yaml",
            ["targets.solar.weather_variable=does_not_exist"],
        )


def test_overlapping_train_and_test_years_rejected() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        Settings.load(REPO_ROOT / "configs" / "default.yaml", ["split.test_years=[2023, 2024]"])


def test_cli_overrides_parse_yaml_values() -> None:
    raw = {"targets": {"solar": {"training": {"epochs": 15}}}}
    apply_overrides(raw, ["targets.solar.training.epochs=3", "split.val_strategy=tail"])
    assert raw["targets"]["solar"]["training"]["epochs"] == 3
    assert raw["split"]["val_strategy"] == "tail"


def test_override_requires_equals_sign() -> None:
    with pytest.raises(ValueError, match="section.key=value"):
        apply_overrides({}, ["no_equals_here"])


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = deep_merge(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}, "d": 3}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


def test_roundtrip_through_yaml(default_settings: Settings) -> None:
    again = Settings.from_yaml_text(default_settings.to_yaml())
    assert again == default_settings
