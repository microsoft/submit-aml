import re
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

from azure.ai.ml import Input
from azure.ai.ml import MLClient
from azure.ai.ml import MpiDistribution
from azure.ai.ml import PyTorchDistribution
from azure.ai.ml import command as instantiate_command
from azure.ai.ml.entities import Job
from azure.ai.ml.entities._job.sweep.search_space import SweepDistribution
from azure.ai.ml.sweep import Choice
from azure.identity import AzureCliCredential
from azure.identity import ManagedIdentityCredential
from rich.console import Console

from .command import TypeServices
from .command import add_service_for_debugging
from .command import add_service_for_tensorboard
from .command import add_sweep_to_inputs_and_script_args
from .command import build_command
from .command import build_debug_command
from .command import log_command
from .config import get_default
from .config import resolve_workspace_config
from .data import TypeOptionalStrList
from .data import build_command_inputs
from .data import build_command_outputs
from .defaults import DEFAULT_SWEEP_ALGORITHM
from .environment import add_profiler_env_variables
from .environment import infer_environment
from .environment import log_environment_variables
from .logger import logger
from .logger import suppress_azure_warnings
from .paths import get_cwd
from .progress import report_time

TypeInputsDict = dict[str, Input | Choice]
_MAX_SWEEP_DESCRIPTION_LENGTH = 511


class CredentialType(StrEnum):
    """Credential type used to authenticate with Azure ML."""

    CLI = "cli"
    """Azure CLI credential of the usercurrently logged-in user."""
    MSI = "msi"
    """Managed Identity assigned to the Azure resource."""


def get_client(
    subscription_id: str | None = None,
    resource_group: str | None = None,
    workspace_name: str | None = None,
    credential_type: CredentialType = CredentialType.CLI,
) -> MLClient:
    """Create and return an Azure ML client.

    Args:
        subscription_id: Azure subscription ID.
        resource_group: Azure resource group name.
        workspace_name: Azure Machine Learning workspace name.
        credential_type: Credential type to use for authentication.

    Returns:
        An authenticated ``MLClient`` instance.
    """
    if credential_type is CredentialType.MSI:
        credential = ManagedIdentityCredential()
        logger.info("Using Managed Identity credential for authentication.")
    else:
        credential = AzureCliCredential(process_timeout=30)
        logger.info("Using Azure CLI credential for authentication.")
    ml_client = MLClient(
        credential,
        subscription_id,
        resource_group,
        workspace_name,
    )
    return ml_client


def setup(
    source_dir: Path | None,
    project_dir: Path | None,
    script_path: Path | str | None,
    subscription_id: str | None,
    resource_group: str | None,
    workspace_name: str | None,
    description: str | None,
    num_gpus: int | None,
    num_nodes: int,
    experiment_name: str | None,
    credential_type: CredentialType = CredentialType.CLI,
) -> tuple[
    Path,
    Path,
    str,
    MLClient,
    str,
    int | None,
    PyTorchDistribution | MpiDistribution | None,
    str | None,
]:
    """Set up multiple variables needed for the job submission.

    Args:
        source_dir: Directory where the source code is located.
        project_dir: Directory where the Python project is located.
        script_path: Script to be executed.
        subscription_id: Azure subscription ID.
        resource_group: Azure resource group name.
        workspace_name: Azure Machine Learning workspace name.
        description: Description of the job.
        num_gpus: Number of GPUs per node (if applicable).
        num_nodes: Number of nodes to use for the job.
        experiment_name: Name of the experiment.
        credential_type: Credential type to use for authentication.

    Returns:
        source_dir: Resolved source directory.
            Current working directory if not provided.
        project_dir: Resolved project directory.
            Current working directory if not provided.
        script_path: Path to the script relative to the source directory.
        ml_client: Azure Machine Learning client.
        description: Description of the job. Command line arguments if not provided.
        instance_count: Number of nodes to use for the job.
        distribution: Distribution strategy for the job (PyTorch or MPI).
    """
    if source_dir is None:
        source_dir = get_cwd()
        msg = (
            f"Source directory not provided. Using the current directory: {source_dir}"
        )
    else:
        source_dir = source_dir.resolve()
        msg = f"Source directory: {source_dir}"
    logger.info(msg)

    ml_client = get_client(
        subscription_id,
        resource_group,
        workspace_name,
        credential_type=credential_type,
    )

    if script_path is None:
        script_path = ""
    else:
        script_path = Path(script_path).resolve()
        script_path = str(script_path.relative_to(source_dir))

    if project_dir is None:
        project_dir = get_cwd()
        msg = (
            "Project directory not provided."
            f" Using the current directory: {project_dir}"
        )
    else:  # assume we're using `uv run` inside the prefix for now
        project_dir = project_dir.resolve()
        msg = f"Project directory: {project_dir}"
    logger.info(msg)

    if description is None:
        description = " ".join(sys.argv)

    instance_count = None
    distribution: PyTorchDistribution | MpiDistribution | None = None
    if num_gpus is None:
        instance_count = num_nodes
        distribution = MpiDistribution()
        logger.info(f'Using "MPI" distribution with {num_nodes} nodes.')
    else:
        instance_count = num_nodes
        distribution = PyTorchDistribution(
            process_count_per_instance=num_gpus,
        )
        logger.info(
            f'Using "PyTorch" distribution with {num_nodes} nodes and '
            f"{num_gpus} GPUs per node."
        )

    experiment_name = _sanitize_experiment_name(experiment_name)

    return (
        source_dir,
        project_dir,
        script_path,
        ml_client,
        description,
        instance_count,
        distribution,
        experiment_name,
    )


def _sanitize_experiment_name(name: str | None) -> str | None:
    """Sanitize the input string to conform to Azure ML experiment naming conventions.

    Azure ML gets very angry when an experiment name contains spaces: it reacts with a
    useless and confusing `HttpResponseError: [...] Bad Request` message.

    This function replaces invalid characters with underscores and collapses multiple
    underscores, and issues a warning if the name was modified.
    """
    if name is None:
        return name
    fixed = re.sub(r"_+", "_", re.sub(r"[^\w-]+", "_", name))
    if fixed != name:
        logger.warning(f'Experiment name changed from "{name}" to "{fixed}"')
    return fixed


def _submit(
    command_job: Job,
    ml_client: MLClient,
    *,
    dry_run: bool = False,
    wait_for_completion: bool = False,
) -> Job | None:
    if dry_run:
        msg = "Dry run mode is enabled. The job won't be submitted."
        logger.warning(msg)
        return None

    start_msg = "Submitting job to Azure Machine Learning..."
    end_msg = "Job submitted successfully"
    with report_time(start_msg, end_msg):
        returned_job = ml_client.create_or_update(command_job)

    logger.info(f'Run ID: "{returned_job.name}"')

    if returned_job.display_name is not None:
        logger.info(f'Display name: "{returned_job.display_name}"')

    logger.info("Studio URL:")
    assert returned_job.services is not None
    url = returned_job.services["Studio"].endpoint
    # Log the run URL. We use this instead of the logger so the URL is clickable and
    # not split over multiple lines.
    # See https://github.com/Textualize/rich/issues/886#issuecomment-756406589
    # for more details.
    Console().print(url, style=f"link {url}")

    if wait_for_completion:
        logger.info("Starting logs streaming...")
        assert returned_job.name is not None
        ml_client.jobs.stream(returned_job.name)

    return returned_job


def submit_to_aml(
    *,
    aml_environment: str | None = None,
    base_docker_image: str = get_default("docker_image"),
    build_docker_context: bool = True,
    command_prefix: str = get_default("command_prefix"),
    compute_target: str | None = get_default("compute_target"),
    conda_env_file: Path | None = None,
    credential_type: CredentialType = CredentialType.CLI,
    datasets_download: TypeOptionalStrList = None,
    datasets_mount: TypeOptionalStrList = None,
    datasets_output: TypeOptionalStrList = None,
    debug: bool = False,
    dependency_groups: list[str] | None = None,
    description: str | None = None,
    docker_run: str | None = None,
    docker_shared_memory_gb: int = get_default("docker_shared_memory_gb"),
    dry_run: bool = False,
    enable_profiler: bool = False,
    enable_tensorboard: bool = get_default("enable_tensorboard"),
    environment_variables: dict[str, Any] | None = None,
    executable: str = get_default("executable"),
    experiment_name: str | None = None,
    num_gpus: int | None = None,
    num_nodes: int = get_default("num_nodes"),
    only_environment: bool = False,
    optional_dependencies: list[str] | None = None,
    project_dir: Path | None = None,
    resource_group: str | None = None,
    run_name: str | None = None,
    services: TypeServices | None = None,
    script_args: list[str] | None = None,
    script_path: Path | str | None = None,
    source_dir: Path | None = None,
    subscription_id: str | None = None,
    sweep_inputs: dict[str, SweepDistribution] | None = None,
    sweep_max_concurrent_trials: int | None = None,
    sweep_prefix: str | None = None,
    sweep_sampling_algorithm: str = DEFAULT_SWEEP_ALGORITHM,
    tags: dict[str, str] | None = None,
    tensorboard_dir: Path = get_default("tensorboard_dir"),
    wait_for_completion: bool = False,
    workspace_name: str | None = get_default("default_workspace"),
) -> Job | None:
    suppress_azure_warnings()

    # Resolve missing subscription_id / resource_group from workspace profiles
    if workspace_name is not None and (
        subscription_id is None or resource_group is None
    ):
        ws_config = resolve_workspace_config(workspace_name)
        if ws_config:
            if subscription_id is None and "subscription_id" in ws_config:
                subscription_id = ws_config["subscription_id"]
                logger.info(
                    'Resolved subscription_id from workspace profile "{}"',
                    workspace_name,
                )
            if resource_group is None and "resource_group" in ws_config:
                resource_group = ws_config["resource_group"]
                logger.info(
                    'Resolved resource_group from workspace profile "{}"',
                    workspace_name,
                )

    # Validate required Azure ML connection parameters
    missing = []
    if workspace_name is None:
        missing.append("--workspace")
    if subscription_id is None:
        missing.append("--subscription")
    if resource_group is None:
        missing.append("--resource-group")
    if missing:
        flags = ", ".join(missing)
        msg = (
            f"Missing required Azure ML configuration: {flags}."
            " Provide them as CLI flags, set up a workspace profile in"
            f" ~/.config/submit-aml/config.toml, or set the corresponding"
            " SUBMIT_AML_ environment variables."
        )
        raise SystemExit(msg)

    if compute_target is None:
        msg = (
            "Missing required compute target (--compute-target)."
            " Provide it as a CLI flag, in the config file under"
            " [compute].compute_target, or as SUBMIT_AML_COMPUTE_TARGET."
        )
        raise SystemExit(msg)

    if conda_env_file is not None:
        if build_docker_context:
            raise ValueError(
                "Cannot use --conda-env-file together with --build-context. "
                "Set --no-build-context when using conda environments."
            )
        if aml_environment is not None:
            raise ValueError(
                "Cannot use --conda-env-file together with --aml-environment. "
                "Choose either conda environment file or existing AML environment."
            )
        if dependency_groups is not None or optional_dependencies is not None:
            raise ValueError(
                "Cannot use --conda-env-file together with"
                " --dependency-group or --extra. "
                "Conda environments manage their own"
                " dependencies."
            )
    (
        source_dir,
        project_dir,
        script_path,
        ml_client,
        description,
        instance_count,
        distribution,
        experiment_name,
    ) = setup(
        source_dir,
        project_dir,
        script_path,
        subscription_id,
        resource_group,
        workspace_name,
        description,
        num_gpus,
        num_nodes,
        experiment_name,
        credential_type=credential_type,
    )

    environment = infer_environment(
        ml_client=ml_client,
        project_dir=project_dir,
        base_docker_image=base_docker_image,
        dependency_groups=dependency_groups,
        optional_dependencies=optional_dependencies,
        aml_environment=aml_environment,
        build_docker_context=build_docker_context,
        conda_env_file=conda_env_file,
        docker_run=docker_run,
        dry_run=dry_run,
    )
    if only_environment:
        msg = "The environment build has been submitted. No job will be submitted."
        logger.warning(msg)
        return None

    # Build command that will be run
    if project_dir != source_dir:
        relative_project_dir = project_dir.relative_to(source_dir)
        command_prefix += f" --project {relative_project_dir}"

    if services is None:
        services = {}
    if debug:
        logger.warning(
            "Debugging mode is enabled. The script won't start until the debugger is"
            " attached"
        )
        command_prefix, executable = build_debug_command(command_prefix, executable)
        add_service_for_debugging(services)
    entry_command = build_command(
        command_prefix,
        executable,
        script_path,
    )
    if script_args is None:
        script_args = []
    # TensorBoard
    if enable_tensorboard:
        add_service_for_tensorboard(services, tensorboard_dir)

    # Data
    inputs = build_command_inputs(ml_client, datasets_download, datasets_mount)
    outputs = build_command_outputs(datasets_output)

    # Sweep jobs
    is_sweep = sweep_inputs is not None and len(sweep_inputs) > 0
    if is_sweep:
        assert sweep_inputs is not None  # not sure why mypy complains without this
        inputs, script_args = add_sweep_to_inputs_and_script_args(
            sweep_inputs,
            inputs,
            script_args,
            sweep_prefix=sweep_prefix,
        )
        # Apparently, "The field Description must be a string with a maximum length of
        # 511" for sweep jobs
        if len(description) > _MAX_SWEEP_DESCRIPTION_LENGTH:
            logger.warning(
                f"Description is too long for a sweep job. "
                f"Truncating to {_MAX_SWEEP_DESCRIPTION_LENGTH} characters."
            )
            suffix = "... (truncated)"
            limit = _MAX_SWEEP_DESCRIPTION_LENGTH - len(suffix)
            description = description[:limit] + suffix
    log_command(entry_command, script_args)

    # Default environment and tags
    if environment_variables is None:
        environment_variables = {}

    if tags is None:
        tags = {}

    # Profiler
    if enable_profiler:
        add_profiler_env_variables(environment_variables)

    log_environment_variables(environment_variables)

    command_string = entry_command
    if script_args:
        command_string = f"{command_string} {' '.join(script_args)}"

    # Create the command job
    job_to_submit = instantiate_command(
        code=source_dir,
        command=command_string,
        compute=compute_target,
        description=description,
        display_name=run_name,
        distribution=distribution,
        environment=environment,
        environment_variables=environment_variables,
        experiment_name=experiment_name,
        inputs=inputs,
        instance_count=instance_count,
        outputs=outputs,
        services=services,
        shm_size=f"{docker_shared_memory_gb}g",
        tags=tags,
    )
    if is_sweep:
        job_to_submit = job_to_submit.sweep(
            compute=compute_target,
            sampling_algorithm=sweep_sampling_algorithm,
            primary_metric="dummy_metric",
            goal="Maximize",
        )
        if sweep_max_concurrent_trials is not None:
            job_to_submit.set_limits(max_concurrent_trials=sweep_max_concurrent_trials)
        job_to_submit.experiment_name = experiment_name  # needed?

    job = _submit(
        job_to_submit,
        ml_client,
        dry_run=dry_run,
        wait_for_completion=wait_for_completion,
    )

    return job
