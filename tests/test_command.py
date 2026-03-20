"""Tests for command-building helpers."""

from __future__ import annotations

import pytest

from submit_aml.command import _parse_sweep_arg
from submit_aml.command import _parse_value_string
from submit_aml.command import build_command
from submit_aml.command import build_debug_command
from submit_aml.command import get_sweep_inputs_from_args
from submit_aml.command import sanitize_input_name

# ---------------------------------------------------------------------------
# build_command (from doctest examples)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prefix", "executable", "script", "expected"),
    [
        ("uv run", "python", "script.py", "uv run python script.py"),
        ("", "nvidia-smi", "", "nvidia-smi"),
        ("uv run --with pyright", "pyright", "", "uv run --with pyright pyright"),
    ],
    ids=["full", "no-prefix-no-script", "no-script"],
)
def test_build_command(
    prefix: str,
    executable: str,
    script: str,
    expected: str,
) -> None:
    """build_command assembles prefix + executable + script correctly."""
    assert build_command(prefix, executable, script) == expected


# ---------------------------------------------------------------------------
# build_debug_command (from doctest examples)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prefix", "executable", "expected_prefix", "expected_exec"),
    [
        (
            "uv run",
            "python",
            "uv run --with debugpy",
            "python -m debugpy --listen localhost:5678 --wait-for-client",
        ),
        (
            "",
            "python",
            "pip install debugpy &&",
            "python -m debugpy --listen localhost:5678 --wait-for-client",
        ),
    ],
    ids=["uv-run", "bare-python"],
)
def test_build_debug_command(
    prefix: str,
    executable: str,
    expected_prefix: str,
    expected_exec: str,
) -> None:
    """build_debug_command injects debugpy into prefix/executable."""
    new_prefix, new_exec = build_debug_command(prefix, executable)
    assert new_prefix == expected_prefix
    assert new_exec == expected_exec


def test_build_debug_command_custom_port() -> None:
    """Custom port is forwarded to the debugpy command."""
    _, executable = build_debug_command("uv run", "python", port=9999)
    assert "localhost:9999" in executable


def test_build_debug_command_non_python_exits() -> None:
    """Non-python executable should cause sys.exit."""
    with pytest.raises(SystemExit):
        build_debug_command("", "bash")


# ---------------------------------------------------------------------------
# sanitize_input_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("model/unet", "model_unet"),
        ("trainer.max_epochs", "trainer_max_epochs"),
        ("+override", "override"),
        ("simple_name", "simple_name"),
    ],
    ids=["slash", "dot", "plus-prefix", "no-change"],
)
def test_sanitize_input_name(raw: str, expected: str) -> None:
    """sanitize_input_name replaces /, . and strips leading +."""
    assert sanitize_input_name(raw) == expected


# ---------------------------------------------------------------------------
# _parse_value_string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", 42),
        ("3.14", 3.14),
        ("1.0e-2", 0.01),
        ("'hello'", "hello"),
        ('"world"', "world"),
        ("simple", "simple"),
    ],
    ids=["int", "float", "scientific", "single-quoted", "double-quoted", "unquoted"],
)
def test_parse_value_string(text: str, expected: int | float | str) -> None:
    """_parse_value_string returns int, float, or str as appropriate."""
    assert _parse_value_string(text) == expected


def test_parse_value_string_raises_on_invalid() -> None:
    """_parse_value_string raises ValueError on unrecognised patterns."""
    with pytest.raises(ValueError, match="Cannot convert"):
        _parse_value_string("foo bar!")


# ---------------------------------------------------------------------------
# _parse_sweep_arg (from doctest examples)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("seed=[0, 1, 2]", ("seed", "", "seed", [0, 1, 2])),
        (
            "model/unet=['tiny', 'small']",
            ("model_unet", "", "model/unet", ["tiny", "small"]),
        ),
        (
            "+trainer.max_epochs=[10, 20]",
            ("trainer_max_epochs", "+", "trainer.max_epochs", [10, 20]),
        ),
        (
            "model.learning_rate=[1.0e-2, 2.0e-2]",
            ("model_learning_rate", "", "model.learning_rate", [0.01, 0.02]),
        ),
    ],
    ids=["simple-ints", "slash-name", "plus-prefix", "scientific-floats"],
)
def test_parse_sweep_arg(
    arg: str,
    expected: tuple[str, str, str, list],
) -> None:
    """_parse_sweep_arg returns (sanitized, prefix, raw, values)."""
    assert _parse_sweep_arg(arg) == expected


def test_parse_sweep_arg_invalid_raises() -> None:
    """Invalid sweep syntax raises ValueError."""
    with pytest.raises(ValueError, match="Invalid sweep argument"):
        _parse_sweep_arg("badformat")


# ---------------------------------------------------------------------------
# get_sweep_inputs_from_args
# ---------------------------------------------------------------------------


def test_get_sweep_inputs_none() -> None:
    """None input returns an empty dict."""
    assert get_sweep_inputs_from_args(None) == {}


def test_get_sweep_inputs_single() -> None:
    """A single sweep arg produces one key in the result."""
    result = get_sweep_inputs_from_args(["seed=[0, 1]"])
    assert "seed" in result


def test_get_sweep_inputs_multiple() -> None:
    """Multiple sweep args produce multiple keys."""
    result = get_sweep_inputs_from_args(["seed=[0, 1]", "+lr=[1e-3, 2e-3]"])
    assert "seed" in result
    assert "+lr" in result
