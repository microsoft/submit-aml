"""Tests for the CLI entry-point."""

from __future__ import annotations

from typer.testing import CliRunner

from submit_aml.__main__ import app

runner = CliRunner()


def test_app_is_importable() -> None:
    """The Typer app object can be imported."""
    assert app is not None


def test_help_exits_zero() -> None:
    """``submit-aml --help`` exits with code 0."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_no_args_shows_missing_config_error() -> None:
    """Running without args or config gives a clear error about missing config."""
    result = runner.invoke(app, [])
    assert "Missing required Azure ML configuration" in result.output
    assert "--workspace" in result.output
    assert "--subscription" in result.output
    assert "--resource-group" in result.output
