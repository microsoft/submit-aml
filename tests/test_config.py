"""Tests for the layered configuration system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from submit_aml.config import _coerce
from submit_aml.config import generate_template_config
from submit_aml.config import get_config
from submit_aml.config import resolve_workspace_config

# ---------------------------------------------------------------------------
# get_config – defaults
# ---------------------------------------------------------------------------


def test_get_config_returns_defaults_without_file_or_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """get_config() should fall back to package defaults.

    Verifies behavior when no config file or env vars exist.
    """
    # Point CONFIG_PATH to a non-existent file so the TOML layer is skipped.
    monkeypatch.setattr(
        "submit_aml.config.CONFIG_PATH",
        tmp_path / "nonexistent.toml",
    )
    # Clear any SUBMIT_AML_ env vars that may leak from the host.
    for key in list(
        monkeypatch._env_patch if hasattr(monkeypatch, "_env_patch") else []
    ):  # noqa: SLF001
        pass  # monkeypatch handles cleanup automatically
    for var in list(k for k in __import__("os").environ if k.startswith("SUBMIT_AML_")):
        monkeypatch.delenv(var, raising=False)

    # get_config is cached – bust the cache for an isolated test.
    get_config.cache_clear()

    cfg = get_config()
    assert cfg["num_nodes"] == 1
    assert cfg["default_workspace"] is None
    assert cfg["executable"] == "python"
    assert isinstance(cfg["tensorboard_dir"], Path)


# ---------------------------------------------------------------------------
# get_config – TOML loading
# ---------------------------------------------------------------------------


def test_get_config_loads_toml(
    monkeypatch: pytest.MonkeyPatch,
    toml_config_file: Path,
) -> None:
    """Values present in the TOML file override package defaults."""
    monkeypatch.setattr("submit_aml.config.CONFIG_PATH", toml_config_file)
    for var in list(k for k in __import__("os").environ if k.startswith("SUBMIT_AML_")):
        monkeypatch.delenv(var, raising=False)
    get_config.cache_clear()

    cfg = get_config()
    assert cfg["default_workspace"] == "from-toml-ws"
    assert cfg["num_nodes"] == 4
    # Keys not in the file should still have defaults.
    assert cfg["executable"] == "python"


# ---------------------------------------------------------------------------
# get_config – env-var override
# ---------------------------------------------------------------------------


def test_get_config_env_var_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An environment variable should override the package default."""
    monkeypatch.setattr(
        "submit_aml.config.CONFIG_PATH",
        tmp_path / "nonexistent.toml",
    )
    monkeypatch.setenv("SUBMIT_AML_NUM_NODES", "8")
    get_config.cache_clear()

    cfg = get_config()
    assert cfg["num_nodes"] == 8


# ---------------------------------------------------------------------------
# get_config – precedence env > TOML > default
# ---------------------------------------------------------------------------


def test_get_config_precedence(
    monkeypatch: pytest.MonkeyPatch,
    toml_config_file: Path,
) -> None:
    """Env vars beat TOML, TOML beats defaults."""
    monkeypatch.setattr("submit_aml.config.CONFIG_PATH", toml_config_file)
    monkeypatch.setenv("SUBMIT_AML_NUM_NODES", "16")
    get_config.cache_clear()

    cfg = get_config()
    # TOML says 4, env says 16 → env wins
    assert cfg["num_nodes"] == 16
    # TOML says "from-toml-ws", no env → TOML wins
    assert cfg["default_workspace"] == "from-toml-ws"


# ---------------------------------------------------------------------------
# _coerce
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "reference", "expected"),
    [
        ("true", True, True),
        ("1", True, True),
        ("yes", True, True),
        ("false", True, False),
        ("0", True, False),
        ("no", True, False),
        ("42", 1, 42),
        ("/tmp/dir", Path("/ref"), Path("/tmp/dir")),
        ("hello", "ref", "hello"),
        ("anything", None, "anything"),
    ],
    ids=[
        "bool-true",
        "bool-1",
        "bool-yes",
        "bool-false",
        "bool-0",
        "bool-no",
        "int",
        "path",
        "str",
        "none-reference",
    ],
)
def test_coerce(value: str, reference: Any, expected: Any) -> None:
    """_coerce should cast env-var strings to the type of the reference default."""
    assert _coerce(value, reference) == expected


# ---------------------------------------------------------------------------
# generate_template_config
# ---------------------------------------------------------------------------


def test_generate_template_config_is_valid_toml() -> None:
    """The generated template must be parseable as TOML."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    text = generate_template_config()
    # All lines are comments or blank, so the parsed result is empty –
    # but parsing must not raise.
    data = tomllib.loads(text)
    assert isinstance(data, dict)


def test_generate_template_config_contains_sections() -> None:
    """The template should mention every TOML section used by the registry."""
    text = generate_template_config()
    for section in (
        "compute",
        "environment",
        "command",
        "tensorboard",
    ):
        assert f"[{section}]" in text
    assert "[azure]" not in text
    assert "default_workspace" in text


def test_generate_template_config_contains_workspace_profiles() -> None:
    """The template should include commented-out workspace profile examples."""
    text = generate_template_config()
    assert "workspaces" in text
    assert "Workspace profiles" in text


# ---------------------------------------------------------------------------
# resolve_workspace_config
# ---------------------------------------------------------------------------


def test_resolve_workspace_config_matching_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A matching workspace profile returns subscription_id and resource_group."""
    config = tmp_path / "config.toml"
    config.write_text(
        '[workspaces.my-ws]\nsubscription_id = "ws-sub-id"\nresource_group = "ws-rg"\n'
    )
    monkeypatch.setattr("submit_aml.config.CONFIG_PATH", config)

    result = resolve_workspace_config("my-ws")
    assert result == {
        "subscription_id": "ws-sub-id",
        "resource_group": "ws-rg",
    }


def test_resolve_workspace_config_no_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-matching workspace name returns an empty dict."""
    config = tmp_path / "config.toml"
    config.write_text('[workspaces.my-ws]\nsubscription_id = "ws-sub-id"\n')
    monkeypatch.setattr("submit_aml.config.CONFIG_PATH", config)

    result = resolve_workspace_config("nonexistent")
    assert result == {}


def test_resolve_workspace_config_no_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Returns empty dict when no config file exists."""
    monkeypatch.setattr(
        "submit_aml.config.CONFIG_PATH",
        tmp_path / "nonexistent.toml",
    )
    result = resolve_workspace_config("anything")
    assert result == {}


def test_resolve_workspace_config_partial_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A profile with only subscription_id returns only that key."""
    config = tmp_path / "config.toml"
    config.write_text('[workspaces.partial]\nsubscription_id = "only-sub"\n')
    monkeypatch.setattr("submit_aml.config.CONFIG_PATH", config)

    result = resolve_workspace_config("partial")
    assert result == {"subscription_id": "only-sub"}
    assert "resource_group" not in result
