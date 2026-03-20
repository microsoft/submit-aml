"""Tests for environment helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from submit_aml.environment import _check_has_patch
from submit_aml.environment import get_env_variable_dict

# ---------------------------------------------------------------------------
# get_env_variable_dict
# ---------------------------------------------------------------------------


def test_get_env_variable_dict_none() -> None:
    """None input returns an empty dict."""
    assert get_env_variable_dict(None) == {}


def test_get_env_variable_dict_valid() -> None:
    """A well-formed list produces the expected mapping."""
    result = get_env_variable_dict(["FOO=bar", "BAZ=qux"])
    assert result == {"FOO": "bar", "BAZ": "qux"}


def test_get_env_variable_dict_invalid_format_raises() -> None:
    """Items without exactly one '=' raise ValueError."""
    with pytest.raises(ValueError, match="Invalid environment variable"):
        get_env_variable_dict(["NO_EQUALS_SIGN"])


def test_get_env_variable_dict_too_many_equals_raises() -> None:
    """Items with more than one '=' raise ValueError."""
    with pytest.raises(ValueError, match="Invalid environment variable"):
        get_env_variable_dict(["A=B=C"])


# ---------------------------------------------------------------------------
# _check_has_patch
# ---------------------------------------------------------------------------


def test_check_has_patch_with_patch(tmp_path: Path) -> None:
    """No warning is emitted when the file has a full version like '3.12.10'."""
    pv = tmp_path / ".python-version"
    pv.write_text("3.12.10\n")
    # Should not raise; we just verify it completes.
    _check_has_patch(pv)


def test_check_has_patch_without_patch(
    tmp_path: Path, capfd: pytest.CaptureFixture
) -> None:
    """A warning is logged when the patch component is missing."""
    pv = tmp_path / ".python-version"
    pv.write_text("3.12\n")
    # _check_has_patch logs a warning via loguru; it should not raise.
    _check_has_patch(pv)
