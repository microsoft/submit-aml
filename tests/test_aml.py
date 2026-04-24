"""Tests for AML helper functions."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from submit_aml.aml import CredentialType
from submit_aml.aml import _sanitize_experiment_name
from submit_aml.aml import get_client


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


# --- get_client credential dispatch ---


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
    """CredentialType.MSI uses ManagedIdentityCredential."""
    get_client("sub", "rg", "ws", credential_type=CredentialType.MSI)
    mock_msi_cred.assert_called_once()  # type: ignore[union-attr]
