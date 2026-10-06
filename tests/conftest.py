from __future__ import annotations

from pathlib import Path

import pytest

from renewcast.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def default_settings() -> Settings:
    return Settings.load(REPO_ROOT / "configs" / "default.yaml")
