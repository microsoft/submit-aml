from __future__ import annotations

import re
import sys
import warnings

from azure.ai.ml import Input
from azure.ai.ml import MLClient
from azure.ai.ml import Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.entities._job.sweep.search_space import SweepDistribution
from azure.ai.ml.exceptions import MlException

from .logger import logger
from .progress import report_time

TypeInputsDict = dict[str, Input | SweepDistribution]
TypeOptionalStrList = list[str] | None


def _datastore_uri(datastore: str, path: str) -> str:
    """Build an Azure ML datastore URI for a folder.

    Args:
        datastore: Name of the datastore.
        path: Path to the folder within the datastore.

    Returns:
        A URI of the form `azureml://datastores/<datastore>/paths/<path>`.

    Examples:
        >>> _datastore_uri('mystore', 'exports/reference')
        'azureml://datastores/mystore/paths/exports/reference'
    """
    return f"azureml://datastores/{datastore}/paths/{path}"


def _extract_alias_path_version(string: str) -> tuple[str, str, str | None]:
    """Get alias, data asset path, and data asset version from a string.

    Args:
        string: String of the form `'alias=path:version'` or `'alias=path'`.

    Returns:
        Tuple of alias, path, and version (which may be
        None if version is not provided).

    Examples:
        >>> _extract_alias_path_version('my_data=MIMIC-CXR-V2:2')
        ('my_data', 'MIMIC-CXR-V2', '2')
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

    Examples:
        >>> _extract_alias_datastore_path('my_data=inereyedata/output_dataset')
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
    """Get alias, job ID, and path from a job output string.

    Args:
        string: String of the form `'alias=<job_id>:<path>'`.

    Returns:
        Tuple of alias, job_id, and path.

    Examples:
        >>> _extract_alias_job_path('checkpoint=my_job_123:models/best.pth')
        ('checkpoint', 'my_job_123', 'models/best.pth')
    """
    pattern = r"(?P<alias>[^=]+)=(?P<job_id>[^:]+):(?P<path>.+)"
    match = re.match(pattern, string)
    if match is None:
        message = (
            f'Invalid job output string: "{string}".'
            ' Expected format: "alias=job_id:path".'
        )
        logger.error(message)
        sys.exit(1)
    if match.group("job_id") == "job_dir":
        message = (
            f'Invalid job output string: "{string}".'
            ' The "job_dir:" prefix is no longer used with the job flags;'
            ' use "alias=job_id:path" instead.'
        )
        logger.error(message)
        sys.exit(1)
    return match.group("alias"), match.group("job_id"), match.group("path")


def _input_from_asset(
    ml_client: MLClient,
    string: str,
    mode: str,
) -> tuple[str, Input]:
    """Build an `Input` from a registered data asset string.

    Args:
        ml_client: Client used to resolve the data asset.
        string: String of the form `'alias=name[:version]'`.
        mode: Either `InputOutputModes.DOWNLOAD` or `InputOutputModes.MOUNT`.

    Returns:
        Tuple of alias and the resolved `Input`.

    Raises:
        ValueError: If the data asset cannot be retrieved.
    """
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
            version_desc = "latest" if version is None else version
            msg = (
                f'Error getting data asset with name "{path}"'
                f' and version "{version_desc}"'
            )
            raise ValueError(msg) from e
    return alias, Input(path=data.id, mode=mode)


def _input_from_datastore(string: str, mode: str) -> tuple[str, Input]:
    """Build an `Input` from a datastore-path string.

    Args:
        string: String of the form `'alias=datastore/folder'`.
        mode: Either `InputOutputModes.DOWNLOAD` or `InputOutputModes.MOUNT`.

    Returns:
        Tuple of alias and the resulting `Input`.
    """
    alias, datastore, folder = _extract_alias_datastore_path(string)
    azureml_path = _datastore_uri(datastore, folder)
    logger.info(f'Using datastore path "{azureml_path}"...')
    return alias, Input(path=azureml_path, mode=mode)


def _input_from_job(string: str, mode: str) -> tuple[str, Input]:
    """Build an `Input` from a previous job's output string.

    Args:
        string: String of the form `'alias=<job_id>:<path>'`.
        mode: Either `InputOutputModes.DOWNLOAD` or `InputOutputModes.MOUNT`.

    Returns:
        Tuple of alias and the resulting `Input`.
    """
    alias, job_id, path = _extract_alias_job_path(string)
    azureml_path = _datastore_uri(
        "workspaceartifactstore",
        f"ExperimentRun/dcid.{job_id}/{path}",
    )
    logger.info(f'Using job output path "{azureml_path}"...')
    return alias, Input(path=azureml_path, mode=mode)


def _output_from_datastore(string: str) -> tuple[str, Output]:
    """Build an `Output` that writes to a datastore folder.

    Args:
        string: String of the form `'alias=datastore/folder'`.

    Returns:
        Tuple of alias and the resulting `Output`.
    """
    alias, datastore, folder = _extract_alias_datastore_path(string)
    return alias, Output(path=_datastore_uri(datastore, folder))


def _output_from_asset(string: str) -> tuple[str, Output]:
    """Build an `Output` that registers a data asset.

    The blobs are written to the workspace's default datastore at an
    Azure ML-managed location and registered as a data asset.

    Args:
        string: String of the form `'alias=name[:version]'`. If the version is
            omitted, Azure ML auto-increments it.

    Returns:
        Tuple of alias and the resulting `Output`.
    """
    alias, name, version = _extract_alias_path_version(string)
    output = Output(type=AssetTypes.URI_FOLDER, name=name, version=version)
    return alias, output


# Removal plan for the deprecated data flags (--mount/-m, --download/-d,
# --output/-o), superseded by the explicit-source flags (--{mount,download}-
# {asset,datastore,job} and --output-{datastore,asset}):
#
#   1. Now (1.x): both flag sets work. The legacy flags carry a [DEPRECATED]
#      marker in --help and emit a deprecation warning at runtime (via
#      `_warn_legacy_input` / `_warn_legacy_output`). This is the grace period
#      in which users migrate.
#   2. Before removal: once downstream callers have migrated (grep the known
#      consumer repos / run scripts for `--mount`, `--download`, `--output`,
#      `-m `, `-d `, `-o ` and the `datasets_{mount,download,output}` kwargs of
#      `submit_to_aml`), and the deprecation has shipped in at least one
#      tagged release, schedule removal for the next MAJOR version (2.0.0) per
#      semver, since dropping a CLI flag is a breaking change.
#   3. At removal (2.0.0): delete the `datasets_download`/`datasets_mount`/
#      `output` typer.Options in `__main__.py`, drop the matching
#      `submit_to_aml` parameters and the `legacy_*` branches in
#      `add_inputs`/`add_outputs`, delete the `_legacy_*` helpers and the
#      `_warn_legacy_*` warning helpers, and note the breaking change in the
#      changelog.
#
# Until step 3, keep the legacy flags VISIBLE in --help (the [DEPRECATED]
# marker is how users discover the migration path); only hide them as an
# optional last step in a release immediately preceding removal.

# Replacement input flag (CLI, Python parameter) for each legacy input flag
# base and classified source type, used to tailor the deprecation warning.
_LEGACY_INPUT_FLAGS = {
    "mount": ("--mount", "datasets_mount"),
    "download": ("--download", "datasets_download"),
}


def _warn_legacy_input(
    flag_base: str,
    source: str,
    old_value: str,
    new_value: str,
    stacklevel: int = 2,
) -> None:
    """Warn that a legacy input flag is deprecated, naming the exact fix.

    Emits both a human-facing log line (for CLI users) and a Python
    `DeprecationWarning` (for callers of the library API, e.g. `submit_to_aml`
    or `build_command_inputs`).

    Args:
        flag_base: The legacy flag base, either `'mount'` or `'download'`.
        source: The classified source type (`'asset'`, `'datastore'`, or
            `'job'`), used to pick the per-source replacement flag.
        old_value: The legacy `alias=value` string the user passed.
        new_value: The value to use with the replacement flag (identical to
            `old_value` except for job values, which drop the `job_dir:`
            prefix).
        stacklevel: Stack level for the `DeprecationWarning`, so it points at
            the API caller rather than this helper.
    """
    old_cli, old_param = _LEGACY_INPUT_FLAGS[flag_base]
    new_cli = f"--{flag_base}-{source}"
    new_param = f"{flag_base}_{source}"
    # CLI users see only flags; Python API users see only parameter names.
    cli_message = (
        f"{old_cli} is deprecated and will be removed in a future release."
        f" Replace '{old_cli} {old_value}' with '{new_cli} {new_value}'."
    )
    api_message = (
        f"The '{old_param}' parameter is deprecated and will be removed in a"
        f" future release. Pass {new_value!r} to '{new_param}' instead."
    )
    logger.warning(cli_message)
    warnings.warn(api_message, DeprecationWarning, stacklevel=stacklevel)


def _warn_legacy_output(old_value: str, stacklevel: int = 2) -> None:
    """Warn that a legacy `--output` value is deprecated, naming the exact fix.

    Emits a human-facing log line naming the replacement CLI flag (for CLI
    users) and a Python `DeprecationWarning` naming the replacement parameter
    (for callers of the library API, e.g. `submit_to_aml` or
    `build_command_outputs`).

    Args:
        old_value: The legacy `alias=datastore/folder` string the user passed.
        stacklevel: Stack level for the `DeprecationWarning`, so it points at
            the API caller rather than this helper.
    """
    # CLI users see only flags; Python API users see only parameter names.
    cli_message = (
        "--output is deprecated and will be removed in a future release."
        f" Replace '--output {old_value}' with '--output-datastore {old_value}'."
    )
    api_message = (
        "The 'datasets_output' parameter is deprecated and will be removed in a"
        f" future release. Pass {old_value!r} to 'output_datastore' instead."
    )
    logger.warning(cli_message)
    warnings.warn(api_message, DeprecationWarning, stacklevel=stacklevel)


def _classify_legacy_input(string: str) -> str:
    """Classify a legacy `--mount`/`--download` value by source type.

    Args:
        string: A legacy dataset string.

    Returns:
        One of `'job'`, `'datastore'`, or `'asset'`.

    A right-hand side that starts with `job_dir:`, or that has a `:` before its
    first `/` (the new `job_id:path` form), is a job output. A `/` that comes
    before any `:` signals a datastore folder. Anything else (a bare `name` or
    `name:version`) is a data asset.

    Examples:
        >>> _classify_legacy_input('ckpt=job_dir:job123:out/best.pth')
        'job'
        >>> _classify_legacy_input('ckpt=job123:out/best.pth')
        'job'
        >>> _classify_legacy_input('ref=mystore/exports/reference')
        'datastore'
        >>> _classify_legacy_input('data=MY-DATASET:2')
        'asset'
    """
    if "=" not in string:
        return "asset"
    _, rhs = string.split("=", 1)
    if rhs.startswith("job_dir:"):
        return "job"
    slash = rhs.find("/")
    if slash == -1:
        return "asset"
    colon = rhs.find(":")
    if colon != -1 and colon < slash:
        return "job"
    return "datastore"


def _legacy_input(
    ml_client: MLClient,
    string: str,
    mode: str,
    flag_base: str,
) -> tuple[str, Input]:
    """Route a legacy `--mount`/`--download` value to the right builder.

    Emits a deprecation warning naming the exact replacement flag for the
    value's classified source type.

    Args:
        ml_client: Client used to resolve data assets.
        string: A legacy dataset string.
        mode: Either `InputOutputModes.DOWNLOAD` or `InputOutputModes.MOUNT`.
        flag_base: The legacy flag base, either `'mount'` or `'download'`,
            used to tailor the deprecation warning.

    Returns:
        Tuple of alias and the resulting `Input`.
    """
    source = _classify_legacy_input(string)
    # stacklevel=4: warnings.warn -> _warn_legacy_input -> _legacy_input ->
    # build_command_inputs, so the DeprecationWarning points at the API caller.
    if source == "job":
        # Translate the old "alias=job_dir:<job_id>:<path>" form to the new one.
        translated = string.replace("=job_dir:", "=", 1)
        _warn_legacy_input(flag_base, "job", string, translated, stacklevel=4)
        return _input_from_job(translated, mode)
    if source == "datastore":
        _warn_legacy_input(flag_base, "datastore", string, string, stacklevel=4)
        return _input_from_datastore(string, mode)
    _warn_legacy_input(flag_base, "asset", string, string, stacklevel=4)
    return _input_from_asset(ml_client, string, mode)


def build_command_inputs(
    ml_client: MLClient,
    *,
    mount_asset: list[str] | None = None,
    download_asset: list[str] | None = None,
    mount_datastore: list[str] | None = None,
    download_datastore: list[str] | None = None,
    mount_job: list[str] | None = None,
    download_job: list[str] | None = None,
    legacy_mount: list[str] | None = None,
    legacy_download: list[str] | None = None,
) -> TypeInputsDict:
    """Build the inputs dictionary for a command job.

    Args:
        ml_client: Client used to resolve data assets.
        mount_asset: Data assets to mount, as `'alias=name[:version]'`.
        download_asset: Data assets to download, as `'alias=name[:version]'`.
        mount_datastore: Datastore folders to mount, as `'alias=datastore/folder'`.
        download_datastore: Datastore folders to download, as
            `'alias=datastore/folder'`.
        mount_job: Previous job outputs to mount, as `'alias=<job_id>:<path>'`.
        download_job: Previous job outputs to download, as `'alias=<job_id>:<path>'`.
        legacy_mount: Deprecated `--mount` values, routed by source type.
        legacy_download: Deprecated `--download` values, routed by source type.

    Returns:
        Dictionary of `alias: Input` mappings.
    """
    inputs: TypeInputsDict = {}

    for string in mount_asset or []:
        alias, value = _input_from_asset(ml_client, string, InputOutputModes.MOUNT)
        inputs[alias] = value
    for string in download_asset or []:
        alias, value = _input_from_asset(ml_client, string, InputOutputModes.DOWNLOAD)
        inputs[alias] = value

    for string in mount_datastore or []:
        alias, value = _input_from_datastore(string, InputOutputModes.MOUNT)
        inputs[alias] = value
    for string in download_datastore or []:
        alias, value = _input_from_datastore(string, InputOutputModes.DOWNLOAD)
        inputs[alias] = value

    for string in mount_job or []:
        alias, value = _input_from_job(string, InputOutputModes.MOUNT)
        inputs[alias] = value
    for string in download_job or []:
        alias, value = _input_from_job(string, InputOutputModes.DOWNLOAD)
        inputs[alias] = value

    if legacy_mount:
        for string in legacy_mount:
            alias, value = _legacy_input(
                ml_client, string, InputOutputModes.MOUNT, "mount"
            )
            inputs[alias] = value
    if legacy_download:
        for string in legacy_download:
            alias, value = _legacy_input(
                ml_client, string, InputOutputModes.DOWNLOAD, "download"
            )
            inputs[alias] = value

    return inputs


def build_command_outputs(
    *,
    output_datastore: list[str] | None = None,
    output_asset: list[str] | None = None,
    legacy_output: list[str] | None = None,
) -> dict[str, Output]:
    """Build the outputs dictionary for a command job.

    Args:
        output_datastore: Datastore folders to write to, as
            `'alias=datastore/folder'`.
        output_asset: Data assets to register, as `'alias=name[:version]'`.
        legacy_output: Deprecated `--output` values (datastore folders).

    Returns:
        Dictionary of `alias: Output` mappings.
    """
    outputs: dict[str, Output] = {}

    for string in output_datastore or []:
        alias, value = _output_from_datastore(string)
        outputs[alias] = value
    for string in output_asset or []:
        alias, value = _output_from_asset(string)
        outputs[alias] = value

    if legacy_output:
        for string in legacy_output:
            # stacklevel=3: warnings.warn -> _warn_legacy_output ->
            # build_command_outputs, pointing at the API caller.
            _warn_legacy_output(string, stacklevel=3)
            alias, value = _output_from_datastore(string)
            outputs[alias] = value

    return outputs
