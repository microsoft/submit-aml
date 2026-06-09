from __future__ import annotations

import re
import sys

from azure.ai.ml import Input
from azure.ai.ml import MLClient
from azure.ai.ml import Output
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.entities._job.sweep.search_space import SweepDistribution
from azure.ai.ml.exceptions import MlException

from .logger import logger
from .progress import report_time

TypeInputsDict = dict[str, Input | SweepDistribution]
TypeOptionalStrList = list[str] | None


def _extract_alias_path_version(string: str) -> tuple[str, str, str | None]:
    """Get alias, data asset path, and data asset version from a string.

    Args:
        string: String of the form `'alias=path:version'` or `'alias=path'`.

    Returns:
        Tuple of alias, path, and version (which may be
        None if version is not provided).

    Raises:
        ValueError: If the string is not of the expected format.

    Examples:
        >>> _extract_alias_path_version('my_data=MIMIC-CXR-V2:2')
        ('my_data', 'MIMIC-CXR-V2', 2)
        >>> _extract_alias_path_version('my_data=MIMIC-CXR-V2')
        ('my_data', 'MIMIC-CXR-V2', None)
    """
    pattern_with_version = r"(?P<alias>[^=]+)=(?P<path>[^:]+):(?P<version>.+)"
    pattern_without_version = r"(?P<alias>[^=]+)=(?P<path>[^:]+)"

    match = re.match(pattern_with_version, string)
    if match:
        return match.group("alias"), match.group("path"), str(match.group("version"))

    match = re.match(pattern_without_version, string)
    if match:
        return match.group("alias"), match.group("path"), None

    message = (
        f'Invalid dataset string: "{string}".'
        ' Expected format: "alias=path:version" or "alias=path".'
    )
    logger.error(message)
    sys.exit(1)


def _datastore_uri(datastore: str, path: str) -> str:
    """Build an Azure ML datastore URI for a folder on a datastore.

    Args:
        datastore: Name of the registered datastore.
        path: Folder path within the datastore.

    Returns:
        An `azureml://` URI of the form
        `azureml://datastores/<datastore>/paths/<path>`.
    """
    return f"azureml://datastores/{datastore}/paths/{path}"


def _extract_alias_datastore_path(string: str) -> tuple[str, str, str]:
    """Get alias, datastore name and folder path from a string.

    Args:
        string: String of the form `'alias=datastore_name/folder/in/datastore'`.

    Returns:
        Tuple of alias, datastore and folder.

    Raises:
        ValueError: If the string is not of the expected format.

    Examples:
        >>> get_alias_datastore_path('my_data=inereyedata/output_dataset')
        ('my_data', 'inereyedata', 'output_dataset')
    """
    pattern = r"(?P<alias>[^=]+)=(?P<datastore>[^/]+)/(?P<folder>.+)"
    match = re.match(pattern, string)
    if match is None:
        message = (
            f'Invalid dataset string: "{string}".'
            ' Expected format: "alias=datastore/folder".'
        )
        logger.error(message)
        sys.exit(1)
    return match.group("alias"), match.group("datastore"), match.group("folder")


def _extract_alias_job_path(string: str) -> tuple[str, str, str]:
    """Get alias, job ID, and path from a job directory string.

    Args:
        string: String of the form `'alias=job_dir:<job_id>:<path>'`.

    Returns:
        Tuple of alias, job_id, and path.

    Raises:
        ValueError: If the string is not of the expected format.

    Examples:
        >>> _extract_alias_job_path('checkpoint=job_dir:my_job_123:models/best.pth')
        ('checkpoint', 'my_job_123', 'models/best.pth')
    """
    pattern = r"(?P<alias>[^=]+)=job_dir:(?P<job_id>[^:]+):(?P<path>.+)"
    match = re.match(pattern, string)
    if match is None:
        message = (
            f'Invalid job directory string: "{string}".'
            ' Expected format: "alias=job_dir:job_id:path".'
        )
        raise ValueError(message)
    return match.group("alias"), match.group("job_id"), match.group("path")


def _is_alias_path_version_string(string: str) -> bool:
    try:
        _extract_alias_path_version(string)
        return True
    except ValueError:
        return False


def _is_alias_job_path_string(string: str) -> bool:
    try:
        _extract_alias_job_path(string)
        return True
    except ValueError:
        return False


def _is_alias_datastore_path_string(string: str) -> bool:
    """Return True if the string refers to a raw datastore-path folder.

    A datastore-path string has the form `'alias=datastore/folder'`. It is
    distinguished from a data-asset name (`'alias=name[:version]'`) by the
    presence of a `/` in the right-hand side, and from a job-output directory
    by not starting with `job_dir:`.

    This is intentionally a pure string check: `_extract_alias_datastore_path`
    calls `sys.exit(1)` on a non-match rather than raising, so it cannot be
    wrapped in try/except the way `_is_alias_job_path_string` is.
    """
    if "=" not in string:
        return False
    _, rhs = string.split("=", 1)
    return "/" in rhs and not rhs.startswith("job_dir:")


def build_command_inputs(
    ml_client: MLClient,
    strings_download: list[str] | None,
    strings_mount: list[str] | None,
) -> TypeInputsDict:
    """Get dictionaries data assets to be mounted or downloaded.

    Args:
        strings_download: List of strings to be downloaded. Each is of the form
            `'alias=name[:version]'` (registered data asset),
            `'alias=datastore/folder'` (raw datastore path), or
            `'alias=job_dir:<job_id>:<path>'` (previous job output).
            If `None`, no data assets will be downloaded.
        strings_mount: List of strings to be mounted, in the same forms as
            `strings_download`. If `None`, no data assets will be mounted.
    """
    strings_download = [] if strings_download is None else strings_download
    strings_mount = [] if strings_mount is None else strings_mount
    datasets_download = _get_data_assets(
        ml_client,
        strings_download,
        InputOutputModes.DOWNLOAD,
    )
    datasets_mount = _get_data_assets(
        ml_client,
        strings_mount,
        InputOutputModes.MOUNT,
    )
    return {**datasets_download, **datasets_mount}


def build_command_outputs(
    strings_upload: list[str] | None,
) -> dict[str, Output]:
    """Get outputs for command.

    Args:
        strings_upload: List of strings of the form `'alias=datastore/path/to/dir'` to
            be uploaded. If `None`, no outputs will be returned.
    """
    strings_upload = [] if strings_upload is None else strings_upload
    outputs_dict = {}
    for string in strings_upload:
        alias, datastore, path = _extract_alias_datastore_path(string)
        output = Output(
            path=_datastore_uri(datastore, path),
        )
        outputs_dict[alias] = output
    return outputs_dict


def _get_data_assets(
    ml_client: MLClient,
    datasets: list[str],
    mode: str,
) -> dict[str, Input]:
    """Get data assets from Azure ML.

    Args:
        datasets: List of strings of the form `'alias=path:version'`,
            `'alias=datastore/folder'`, or `'alias=job_dir:<job_id>:<path>'`.
        mode: Either `InputOutputModes.DOWNLOAD` or `InputOutputModes.MOUNT`.

    Returns:
        Dictionary of `alias: Input` mappings.
    """
    inputs = {}
    for string in datasets:
        if _is_alias_job_path_string(string):
            # Handle job directory format
            alias, job_id, path = _extract_alias_job_path(string)
            azureml_path = f"azureml://datastores/workspaceartifactstore/paths/ExperimentRun/dcid.{job_id}/{path}"
            logger.info(f'Using job output path "{azureml_path}"...')
            inputs[alias] = Input(
                path=str(azureml_path),
                mode=mode,
            )
        elif _is_alias_datastore_path_string(string):
            # Handle raw datastore-path folder format
            alias, datastore, folder = _extract_alias_datastore_path(string)
            azureml_path = _datastore_uri(datastore, folder)
            logger.info(f'Using datastore path "{azureml_path}"...')
            inputs[alias] = Input(
                path=azureml_path,
                mode=mode,
            )
        else:
            # Handle regular data asset format
            alias, path, version = _extract_alias_path_version(string)

            if version is None:
                kwargs = {"label": "latest"}
            else:
                kwargs = {"version": version}

            with report_time(
                f'Retrieving data asset "{path}"...',
                f'Retrieved data asset "{path}"',
            ):
                try:
                    data = ml_client.data.get(name=path, **kwargs)
                except MlException as e:
                    msg = (
                        "Error getting data asset with"
                        f' name "{path}"'
                        f' and version "{version}"'
                    )
                    raise ValueError(msg) from e
            inputs[alias] = Input(
                path=data.id,
                mode=mode,
            )
    return inputs
