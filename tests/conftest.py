"""Shared fixtures for submit-aml tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def toml_config_file(tmp_path: Path) -> Path:
    """Write a minimal TOML config file and return its path."""
    config = tmp_path / "config.toml"
    config.write_text(
        'default_workspace = "from-toml-ws"\n\n[compute]\nnum_nodes = 4\n'
    )
    return config
