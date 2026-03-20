"""Tests for AML helper functions."""

from __future__ import annotations

import pytest

from submit_aml.aml import _sanitize_experiment_name


def test_sanitize_none_returns_none() -> None:
    """None input is returned as-is."""
    assert _sanitize_experiment_name(None) is None


def test_sanitize_clean_name_unchanged() -> None:
    """A name with only valid characters is returned unchanged."""
    assert _sanitize_experiment_name("my-experiment_1") == "my-experiment_1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("my experiment", "my_experiment"),
        ("hello world!", "hello_world_"),
        ("a  b", "a_b"),
        ("foo@bar#baz", "foo_bar_baz"),
    ],
    ids=["spaces", "special-char", "double-space", "multiple-specials"],
)
def test_sanitize_replaces_invalid_chars(raw: str, expected: str) -> None:
    """Spaces and special characters are replaced with underscores."""
    assert _sanitize_experiment_name(raw) == expected
