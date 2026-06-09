"""Tests for data-asset parsing helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from azure.ai.ml.constants import InputOutputModes

from submit_aml.data import _extract_alias_datastore_path
from submit_aml.data import _extract_alias_job_path
from submit_aml.data import _extract_alias_path_version
from submit_aml.data import _is_alias_datastore_path_string
from submit_aml.data import build_command_inputs
from submit_aml.data import build_command_outputs

# ---------------------------------------------------------------------------
# _extract_alias_path_version
# ---------------------------------------------------------------------------


def test_extract_alias_path_version_with_version() -> None:
    """Strings of the form 'alias=path:version' are parsed correctly."""
    alias, path, version = _extract_alias_path_version("my_data=MIMIC-CXR-V2:2")
    assert alias == "my_data"
    assert path == "MIMIC-CXR-V2"
    assert version == "2"


def test_extract_alias_path_version_without_version() -> None:
    """Omitting the version yields None."""
    alias, path, version = _extract_alias_path_version("my_data=MIMIC-CXR-V2")
    assert alias == "my_data"
    assert path == "MIMIC-CXR-V2"
    assert version is None


# ---------------------------------------------------------------------------
# _extract_alias_datastore_path
# ---------------------------------------------------------------------------


def test_extract_alias_datastore_path_valid() -> None:
    """'alias=datastore/folder' is parsed into three components."""
    alias, ds, folder = _extract_alias_datastore_path(
        "my_data=inereyedata/output_dataset"
    )
    assert alias == "my_data"
    assert ds == "inereyedata"
    assert folder == "output_dataset"


# ---------------------------------------------------------------------------
# _extract_alias_job_path
# ---------------------------------------------------------------------------


def test_extract_alias_job_path_valid() -> None:
    """'alias=job_dir:job_id:path' is parsed correctly."""
    alias, job_id, path = _extract_alias_job_path(
        "checkpoint=job_dir:my_job_123:models/best.pth"
    )
    assert alias == "checkpoint"
    assert job_id == "my_job_123"
    assert path == "models/best.pth"


def test_extract_alias_job_path_invalid_raises() -> None:
    """Strings not matching the job_dir pattern raise ValueError."""
    with pytest.raises(ValueError, match="Invalid job directory"):
        _extract_alias_job_path("bad_format")


# ---------------------------------------------------------------------------
# build_command_outputs
# ---------------------------------------------------------------------------


def test_build_command_outputs_none() -> None:
    """None input produces an empty dict."""
    assert build_command_outputs(None) == {}


def test_build_command_outputs_valid() -> None:
    """Valid output strings are converted into Output objects."""
    outputs = build_command_outputs(["out_dir=mydatastore/my_dataset"])
    assert "out_dir" in outputs
    output = outputs["out_dir"]
    assert "mydatastore" in output.path
    assert "my_dataset" in output.path


# ---------------------------------------------------------------------------
# _is_alias_datastore_path_string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("ref=mystore/exports/reference", True),
        ("ref=mystore/folder", True),
        ("my_data=MIMIC-CXR-V2", False),
        ("my_data=MIMIC-CXR-V2:2", False),
        ("checkpoint=job_dir:my_job_123:models/best.pth", False),
        ("no_equals_sign", False),
    ],
)
def test_is_alias_datastore_path_string(string: str, expected: bool) -> None:
    """Only 'alias=datastore/folder' strings are recognised as datastore paths."""
    assert _is_alias_datastore_path_string(string) is expected


# ---------------------------------------------------------------------------
# build_command_inputs (datastore-path branch)
# ---------------------------------------------------------------------------


def test_build_command_inputs_datastore_path_mount() -> None:
    """A raw datastore-path string builds an azureml:// Input without AML lookup."""
    ml_client = MagicMock()
    inputs = build_command_inputs(
        ml_client,
        strings_download=None,
        strings_mount=["ref=mystore/exports/reference"],
    )
    assert "ref" in inputs
    ref = inputs["ref"]
    assert ref.path == "azureml://datastores/mystore/paths/exports/reference"
    assert ref.mode == InputOutputModes.MOUNT
    ml_client.data.get.assert_not_called()


def test_build_command_inputs_datastore_path_download() -> None:
    """The datastore-path branch honours the download mode."""
    ml_client = MagicMock()
    inputs = build_command_inputs(
        ml_client,
        strings_download=["ref=mystore/exports/reference"],
        strings_mount=None,
    )
    assert inputs["ref"].mode == InputOutputModes.DOWNLOAD
    ml_client.data.get.assert_not_called()


def test_build_command_inputs_data_asset_routes_to_get() -> None:
    """A 'name:version' string still resolves via ml_client.data.get."""
    ml_client = MagicMock()
    ml_client.data.get.return_value = MagicMock(id="azureml://data-asset-id")
    inputs = build_command_inputs(
        ml_client,
        strings_download=None,
        strings_mount=["my_data=MIMIC-CXR-V2:2"],
    )
    ml_client.data.get.assert_called_once_with(name="MIMIC-CXR-V2", version="2")
    assert inputs["my_data"].path == "azureml://data-asset-id"
