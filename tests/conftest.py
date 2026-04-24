"""Shared fixtures for submit-aml tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Point the CLI at a guaranteed non-existent config file BEFORE ``submit_aml``
# is imported anywhere in the test session, so that Typer option defaults
# (evaluated at import time) do not pick up the developer's real
# ``~/.config/submit-aml/config.toml``. Likewise drop any ``SUBMIT_AML_*``
# variables that may leak from the host shell.
os.environ["SUBMIT_AML_CONFIG"] = "/nonexistent/submit-aml/config.toml"
for _var in [
    k for k in os.environ if k.startswith("SUBMIT_AML_") and k != "SUBMIT_AML_CONFIG"
]:
    del os.environ[_var]


@pytest.fixture
def toml_config_file(tmp_path: Path) -> Path:
    """Write a minimal TOML config file and return its path."""
    config = tmp_path / "config.toml"
    config.write_text(
        'default_workspace = "from-toml-ws"\n\n[compute]\nnum_nodes = 4\n'
    )
    return config
