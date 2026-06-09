"""Tests for data-asset parsing helpers and input/output builders."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.exceptions import MlException

from submit_aml.data import _classify_legacy_input
from submit_aml.data import _datastore_uri
from submit_aml.data import _extract_alias_datastore_path
from submit_aml.data import _extract_alias_job_path
from submit_aml.data import _extract_alias_path_version
from submit_aml.data import _input_from_asset
from submit_aml.data import _input_from_datastore
from submit_aml.data import _input_from_job
from submit_aml.data import _output_from_asset
from submit_aml.data import _output_from_datastore
from submit_aml.data import build_command_inputs
from submit_aml.data import build_command_outputs

# ---------------------------------------------------------------------------
# _datastore_uri
# ---------------------------------------------------------------------------


def test_datastore_uri() -> None:
    """A datastore and path are joined into an azureml:// URI."""
    uri = _datastore_uri("mystore", "exports/reference")
    assert uri == "azureml://datastores/mystore/paths/exports/reference"


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
    """'alias=job_id:path' is parsed correctly (no job_dir: prefix)."""
    alias, job_id, path = _extract_alias_job_path(
        "checkpoint=my_job_123:models/best.pth"
    )
    assert alias == "checkpoint"
    assert job_id == "my_job_123"
    assert path == "models/best.pth"


def test_extract_alias_job_path_invalid_exits() -> None:
    """Strings without a path component exit the process."""
    with pytest.raises(SystemExit):
        _extract_alias_job_path("bad_format")


def test_extract_alias_job_path_rejects_legacy_prefix() -> None:
    """The legacy 'job_dir:' prefix is rejected on the new job flags."""
    with pytest.raises(SystemExit):
        _extract_alias_job_path("ckpt=job_dir:my_job_123:models/best.pth")


# ---------------------------------------------------------------------------
# _classify_legacy_input
# ---------------------------------------------------------------------------


def test_classify_legacy_input_job() -> None:
    """A job_dir: prefix is classified as a job output."""
    assert _classify_legacy_input("ckpt=job_dir:job123:out/best.pth") == "job"


def test_classify_legacy_input_job_new_syntax() -> None:
    """A new-style 'job_id:path' value (colon before slash) is a job output."""
    assert _classify_legacy_input("ckpt=job123:outputs/best.pth") == "job"


def test_classify_legacy_input_datastore() -> None:
    """A slash before any colon signals a datastore path."""
    assert _classify_legacy_input("ref=mystore/exports/reference") == "datastore"


def test_classify_legacy_input_datastore_with_colon_in_folder() -> None:
    """A colon after the first slash is part of the folder, not a job id."""
    assert _classify_legacy_input("ref=mystore/a:b/c") == "datastore"


def test_classify_legacy_input_asset() -> None:
    """A plain name[:version] is classified as a data asset."""
    assert _classify_legacy_input("data=MY-DATASET:2") == "asset"


def test_classify_legacy_input_missing_equals() -> None:
    """A string without '=' falls back to the asset branch."""
    assert _classify_legacy_input("no-equals-here") == "asset"


# ---------------------------------------------------------------------------
# input builders
# ---------------------------------------------------------------------------


def test_input_from_datastore() -> None:
    """A datastore string yields an Input with an azureml:// path and mode."""
    alias, value = _input_from_datastore(
        "ref=mystore/exports/reference",
        InputOutputModes.MOUNT,
    )
    assert alias == "ref"
    assert value.path == "azureml://datastores/mystore/paths/exports/reference"
    assert value.mode == InputOutputModes.MOUNT


def test_input_from_job() -> None:
    """A job string yields an Input pointing at the job's run artifacts."""
    alias, value = _input_from_job(
        "checkpoint=my_job_123:models/best.pth",
        InputOutputModes.DOWNLOAD,
    )
    assert alias == "checkpoint"
    assert "ExperimentRun/dcid.my_job_123/models/best.pth" in value.path
    assert "workspaceartifactstore" in value.path
    assert value.mode == InputOutputModes.DOWNLOAD


def test_input_from_asset_missing_version_reports_latest() -> None:
    """When no version is given, the failure message mentions 'latest'."""
    client = Mock()
    client.data.get.side_effect = MlException(
        message="boom", no_personal_data_message="boom"
    )
    with pytest.raises(ValueError, match='version "latest"'):
        _input_from_asset(client, "data=MY-DATASET", InputOutputModes.MOUNT)


# ---------------------------------------------------------------------------
# output builders
# ---------------------------------------------------------------------------


def test_output_from_datastore() -> None:
    """A datastore string yields an Output with an azureml:// path."""
    alias, output = _output_from_datastore("out_dir=mydatastore/my_dataset")
    assert alias == "out_dir"
    assert output.path == "azureml://datastores/mydatastore/paths/my_dataset"


def test_output_from_asset_with_version() -> None:
    """An asset string registers an Output with name and version."""
    alias, output = _output_from_asset("out_dir=my-results:3")
    assert alias == "out_dir"
    assert output.name == "my-results"
    assert output.version == "3"
    assert output.type == "uri_folder"


def test_output_from_asset_without_version() -> None:
    """Omitting the version leaves it unset for Azure ML to auto-increment."""
    _, output = _output_from_asset("out_dir=my-results")
    assert output.name == "my-results"
    assert output.version is None


# ---------------------------------------------------------------------------
# build_command_inputs
# ---------------------------------------------------------------------------


def test_build_command_inputs_empty() -> None:
    """No arguments produce an empty dict and never touch the client."""
    client = Mock()
    assert build_command_inputs(client) == {}
    client.data.get.assert_not_called()


def test_build_command_inputs_datastore_and_job_skip_client() -> None:
    """Datastore and job inputs are built without calling the client."""
    client = Mock()
    inputs = build_command_inputs(
        client,
        mount_datastore=["ref=mystore/exports/reference"],
        download_job=["ckpt=my_job_123:models/best.pth"],
    )
    assert set(inputs) == {"ref", "ckpt"}
    assert inputs["ref"].mode == InputOutputModes.MOUNT
    assert inputs["ckpt"].mode == InputOutputModes.DOWNLOAD
    client.data.get.assert_not_called()


def test_build_command_inputs_asset_calls_client() -> None:
    """A data-asset input resolves through the client."""
    client = Mock()
    client.data.get.return_value = Mock(id="azureml:resolved-asset:1")
    inputs = build_command_inputs(client, mount_asset=["data=MY-DATASET:2"])
    client.data.get.assert_called_once()
    assert inputs["data"].path == "azureml:resolved-asset:1"


def test_build_command_inputs_legacy_datastore_routes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Legacy --mount datastore strings route to the datastore builder."""
    client = Mock()
    inputs = build_command_inputs(
        client,
        legacy_mount=["ref=mystore/exports/reference"],
    )
    assert inputs["ref"].path == (
        "azureml://datastores/mystore/paths/exports/reference"
    )
    client.data.get.assert_not_called()
    assert "deprecated" in capsys.readouterr().out.lower()


def test_build_command_inputs_legacy_job_routes() -> None:
    """Legacy --download job_dir strings route to the job builder."""
    client = Mock()
    inputs = build_command_inputs(
        client,
        legacy_download=["ckpt=job_dir:my_job_123:models/best.pth"],
    )
    assert "ExperimentRun/dcid.my_job_123/models/best.pth" in inputs["ckpt"].path
    client.data.get.assert_not_called()


def test_build_command_inputs_legacy_job_new_syntax_routes() -> None:
    """Legacy values using the new 'job_id:path' form route to the job builder."""
    client = Mock()
    inputs = build_command_inputs(
        client,
        legacy_mount=["ckpt=my_job_123:models/best.pth"],
    )
    assert "ExperimentRun/dcid.my_job_123/models/best.pth" in inputs["ckpt"].path
    client.data.get.assert_not_called()


def test_build_command_inputs_legacy_asset_calls_client() -> None:
    """Legacy --mount asset strings still resolve through the client."""
    client = Mock()
    client.data.get.return_value = Mock(id="azureml:resolved-asset:1")
    build_command_inputs(client, legacy_mount=["data=MY-DATASET:2"])
    client.data.get.assert_called_once()


def test_build_command_inputs_legacy_asset_warns_mount_asset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A legacy --mount asset value is told to use --mount-asset specifically."""
    client = Mock()
    client.data.get.return_value = Mock(id="azureml:resolved-asset:1")
    build_command_inputs(client, legacy_mount=["my_alias=data_asset"])
    message = " ".join(capsys.readouterr().out.split())
    assert "--mount-asset my_alias=data_asset" in message
    assert "--mount-datastore" not in message
    assert "--mount-job" not in message


def test_build_command_inputs_legacy_datastore_warns_mount_datastore(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A legacy --mount datastore value is told to use --mount-datastore."""
    client = Mock()
    build_command_inputs(client, legacy_mount=["ref=mystore/exports/reference"])
    message = " ".join(capsys.readouterr().out.split())
    assert "--mount-datastore ref=mystore/exports/reference" in message


def test_build_command_inputs_legacy_job_warns_download_job_translated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A legacy --download job value is told to use --download-job, sans prefix."""
    client = Mock()
    build_command_inputs(
        client,
        legacy_download=["ckpt=job_dir:my_job_123:models/best.pth"],
    )
    message = " ".join(capsys.readouterr().out.split())
    assert "--download-job ckpt=my_job_123:models/best.pth" in message
    # The suggested replacement (after "with") drops the legacy job_dir: prefix.
    assert "job_dir:" not in message.split("with", 1)[1]


# ---------------------------------------------------------------------------
# build_command_outputs
# ---------------------------------------------------------------------------


def test_build_command_outputs_empty() -> None:
    """No arguments produce an empty dict."""
    assert build_command_outputs() == {}


def test_build_command_outputs_datastore_and_asset() -> None:
    """Datastore and asset outputs are both built."""
    outputs = build_command_outputs(
        output_datastore=["out_dir=mydatastore/my_dataset"],
        output_asset=["asset_dir=my-results:2"],
    )
    assert outputs["out_dir"].path == (
        "azureml://datastores/mydatastore/paths/my_dataset"
    )
    assert outputs["asset_dir"].name == "my-results"
    assert outputs["asset_dir"].version == "2"


def test_build_command_outputs_legacy_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Legacy --output strings are built and emit a targeted deprecation warning."""
    outputs = build_command_outputs(legacy_output=["out_dir=mydatastore/my_dataset"])
    assert "out_dir" in outputs
    message = " ".join(capsys.readouterr().out.split())
    assert "deprecated" in message.lower()
    assert "--output-datastore out_dir=mydatastore/my_dataset" in message
