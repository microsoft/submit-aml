"""Tests for AML helper functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from submit_aml.aml import CredentialType
from submit_aml.aml import _sanitize_experiment_name
from submit_aml.aml import get_client
from submit_aml.aml import submit_to_aml


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


@patch("submit_aml.aml.MLClient")
@patch("submit_aml.aml.AzureCliCredential")
def test_get_client_default_uses_cli_credential(
    mock_cli_cred: object,
    mock_ml_client: object,
) -> None:
    """Default credential type uses AzureCliCredential."""
    get_client("sub", "rg", "ws")
    mock_cli_cred.assert_called_once_with(process_timeout=30)  # type: ignore[union-attr]


@patch("submit_aml.aml.MLClient")
@patch("submit_aml.aml.ManagedIdentityCredential")
def test_get_client_msi_uses_managed_identity(
    mock_msi_cred: object,
    mock_ml_client: object,
) -> None:
    """CredentialType.MANAGED_IDENTITY uses ManagedIdentityCredential."""
    get_client("sub", "rg", "ws", credential_type=CredentialType.MANAGED_IDENTITY)
    mock_msi_cred.assert_called_once()  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("command_prefix", "expected_in_command"),
    [
        ("uv run", "--project subproject"),
        ("python", None),
    ],
    ids=["uv-run-appends-project", "non-uv-skips-project"],
)
@patch("submit_aml.aml._submit")
@patch("submit_aml.aml.instantiate_command")
@patch("submit_aml.aml.infer_environment")
@patch("submit_aml.aml.setup")
def test_project_flag_only_appended_for_uv_run(
    mock_setup: object,
    mock_infer_env: object,
    mock_instantiate: object,
    mock_submit: object,
    command_prefix: str,
    expected_in_command: str | None,
) -> None:
    """``--project`` is appended only for ``uv run`` prefixes."""
    source_dir = Path("/repo")
    project_dir = source_dir / "subproject"
    mock_setup.return_value = (  # type: ignore[attr-defined]
        source_dir,
        project_dir,
        "run.py",
        object(),  # ml_client
        "description",
        1,  # instance_count
        None,  # distribution
        "experiment",
    )

    submit_to_aml(
        command_prefix=command_prefix,
        compute_target="cpu-cluster",
        script_path="run.py",
        subscription_id="sub",
        resource_group="rg",
        workspace_name="ws",
        dry_run=True,
    )

    command = mock_instantiate.call_args.kwargs["command"]  # type: ignore[attr-defined]
    if expected_in_command is None:
        assert "--project" not in command
    else:
        assert expected_in_command in command


@patch("submit_aml.aml._submit")
@patch("submit_aml.aml.instantiate_command")
@patch("submit_aml.aml.infer_environment")
@patch("submit_aml.aml.setup")
def test_uv_run_prefix_raises_when_project_not_under_source(
    mock_setup: object,
    mock_infer_env: object,
    mock_instantiate: object,
    mock_submit: object,
) -> None:
    """A clear ValueError is raised when project_dir is not inside source_dir."""
    source_dir = Path("/repo")
    project_dir = Path("/other/project")
    mock_setup.return_value = (  # type: ignore[attr-defined]
        source_dir,
        project_dir,
        "run.py",
        object(),  # ml_client
        "description",
        1,  # instance_count
        None,  # distribution
        "experiment",
    )

    with pytest.raises(ValueError, match="must be inside the source directory"):
        submit_to_aml(
            command_prefix="uv run",
            compute_target="cpu-cluster",
            script_path="run.py",
            subscription_id="sub",
            resource_group="rg",
            workspace_name="ws",
            dry_run=True,
        )
