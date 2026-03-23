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


def build_command_inputs(
    ml_client: MLClient,
    strings_download: list[str] | None,
    strings_mount: list[str] | None,
) -> TypeInputsDict:
    """Get dictionaries data assets to be mounted or downloaded.

    Args:
        strings_download: List of strings of the form `'alias=path:version'` to
            be downloaded. If `None`, no data assets will be downloaded.
        strings_mount: List of strings of the form `'alias=path:version'` to
            be mounted. If `None`, no data assets will be mounted.
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
            path=f"azureml://datastores/{datastore}/paths/{path}",
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
        datasets: List of strings of the form `'alias=path:version'` or
            `'alias=job_dir:<job_id>:<path>'`.
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
        else:
            # Handle regular data asset format
            alias, path, version = _extract_alias_path_version(string)

            if version is None:
                kwargs = {"label": "latest"}
            else:
                kwargs = {"version": version}

            logger.info(f'Retrieving data asset "{path}"...')
            try:
                data = ml_client.data.get(name=path, **kwargs)
            except MlException as e:
                msg = (
                    "Error getting data asset with"
                    f' name "{path}"'
                    f' and version "{version}"'
                )
                raise ValueError(msg) from e
            logger.success(f'Found data asset with path "{path}"')
            inputs[alias] = Input(
                path=data.id,
                mode=mode,
            )
    return inputs
