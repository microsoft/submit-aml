# ruff: noqa: B008, C901, FBT001, FBT003, PLR0912, PLR0913, PLR0915

from pathlib import Path
from typing import List  # noqa: UP035
from typing import Optional

import typer
from rich.console import Console

from .aml import submit_to_aml
from .command import get_sweep_inputs_from_args
from .config import get_default
from .environment import get_env_variable_dict
from .logger import logger

PANEL_AZURE = "Azure"
PANEL_COMMAND = "Command"
PANEL_COMPUTE = "Compute resources"
PANEL_DATA = "Data"
PANEL_ENVIRONMENT = "Environment"
PANEL_INTERACTION = "Interaction"


# This syntax is used so that this file can be a console script
app = typer.Typer(
    pretty_exceptions_show_locals=False,
)


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def submit(
    experiment_name: str = typer.Option(
        None,
        "--experiment-name",
        "-e",
        help=(
            "Name of the Azure ML experiment to which the job will be submitted."
            " If not provided, the name of the current directory name will be used."
        ),
        rich_help_panel=PANEL_AZURE,
    ),
    run_name: str = typer.Option(
        None,
        "--run-name",
        "-r",
        help="Display name of the Azure ML run.",
        rich_help_panel=PANEL_AZURE,
    ),
    workspace_name: str = typer.Option(
        get_default("default_workspace"),
        "--workspace",
        help="Name of the Azure ML workspace.",
        rich_help_panel=PANEL_AZURE,
    ),
    resource_group: str = typer.Option(
        None,
        "--resource-group",
        "-g",
        help="Name of the Azure ML resource group.",
        rich_help_panel=PANEL_AZURE,
    ),
    subscription_id: str = typer.Option(
        None,
        "--subscription",
        help="Subscription ID of the workspace.",
        rich_help_panel=PANEL_AZURE,
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        help=(
            "Description for the Azure ML job."
            " If not provided, the local command will be used."
        ),
        rich_help_panel=PANEL_AZURE,
    ),
    compute_target: str = typer.Option(
        get_default("compute_target"),
        "--compute-target",
        "-c",
        help="Name of the Azure ML compute target to run the job on.",
        rich_help_panel=PANEL_COMPUTE,
    ),
    docker_image: str = typer.Option(
        get_default("docker_image"),
        "--docker-image",
        "-i",
        help="Base Docker image to use for the job.",
        rich_help_panel=PANEL_ENVIRONMENT,
    ),
    build_docker_context: bool = typer.Option(
        True,
        "--build-context/--no-build-context",
        help="Whether to build a Docker context from the project directory.",
        rich_help_panel=PANEL_ENVIRONMENT,
    ),
    docker_run: str | None = typer.Option(
        None,
        help="Extra command to run in Docker build before syncing the environment.",
        rich_help_panel=PANEL_ENVIRONMENT,
    ),
    aml_environment: str | None = typer.Option(
        None,
        "--aml-environment",
        help=(
            "Name of an existing Azure ML environment to use"
            " for the job. If provided, the Docker image and"
            " build context arguments will be ignored."
        ),
        rich_help_panel=PANEL_ENVIRONMENT,
    ),
    docker_shared_memory_gb: int = typer.Option(
        get_default("docker_shared_memory_gb"),
        "--shared-memory",
        help="Amount of shared memory for the Docker container (in GB)",
        rich_help_panel=PANEL_COMPUTE,
    ),
    num_nodes: int = typer.Option(
        get_default("num_nodes"),
        "--num-nodes",
        "-n",
        help="Number of nodes to use for the job.",
        rich_help_panel=PANEL_COMPUTE,
    ),
    datasets_download: Optional[List[str]] = typer.Option(  # noqa: UP006, UP007
        None,
        "--download",
        "-d",
        help=(
            "Azure ML dataset or job output folder to download. To download an Azure ML"
            " dataset, the argument should take the form: alias, name and version"
            " of the dataset; for example: 'vindr_dir=VINDR-CXR-V2:1'."
            " If the version is omitted, the last one will be used."
            " To download the output folder of a previous job, the argument should take"
            " the form 'alias=job_dir:<job_id>:<path/in/job/outputs>'; for example:"
            " 'checkpoint=job_dir:crusty_hat_43s6lmvb25:outputs/checkpoint-10000'."
            " The alias can be used to pass input datasets to the script, e.g.,"
            r" '${{inputs.vindr_dir}}' or '${{inputs.checkpoint}}'."
            " This option can be used multiple times."
        ),
        rich_help_panel=PANEL_DATA,
    ),
    datasets_mount: Optional[List[str]] = typer.Option(  # noqa: UP006, UP007
        None,
        "--mount",
        "-m",
        help=(
            "Azure ML dataset or job output folder to mount."
            " For an Azure ML dataset, the alias, name and version should be provided"
            " while for a job output folder, the alias, job ID and path in the job"
            " outputs should be provided. See the --download option for more"
            " information."
        ),
        rich_help_panel=PANEL_DATA,
    ),
    output: Optional[List[str]] = typer.Option(  # noqa: UP006, UP007
        None,
        "--output",
        "-o",
        help=(
            "Alias, datastore and path to folder into which outputs will be written,"
            ' expressed as "alias=datastore/path/to/dir".'
            ' For example: "out_dir=mydatastore/my_dataset".'
            " The alias can be used to pass outputs to the script, e.g.,"
            r' "${{outputs.out_dir}}".'
            " See the example for more information."
            " This option can be used multiple times."
        ),
        rich_help_panel=PANEL_DATA,
    ),
    command_prefix: str = typer.Option(
        get_default("command_prefix"),
        help="Prefix to prepend to the command. For example, `uv run`.",
        rich_help_panel=PANEL_COMMAND,
    ),
    executable: str = typer.Option(
        get_default("executable"),
        help=(
            "The executable, e.g., `python`, `'torchrun --nproc-per-node auto'`,"
            " `bash`, or `nvidia-smi`."
        ),
        rich_help_panel=PANEL_COMMAND,
    ),
    script_path: Path = typer.Option(
        None,
        "--script",
        "-s",
        exists=True,
        help="Path to the script that will be run on Azure ML.",
        rich_help_panel=PANEL_COMMAND,
    ),
    sweep_args: Optional[List[str]] = typer.Option(
        None,
        "--sweep",
        help=(
            "Azure ML hyperparameter for sweep jobs."
            " Examples:"
            ' "seed=[0, 1, 2]",'
            " \"model/unet=['tiny', 'small']\","
            ' "+trainer.max_epochs=[10, 20]",'
            ' "model.learning_rate=[1.0e-4, 2.0e-4]".'
            " If a `--sweep-prefix` is passed, the sweep arguments will be added to the"
            " command with the prefix."
            " The keys are adapted to be compatible with Azure ML Inputs and will be"
            " available as environment variables in the job. For the examples above,"
            " the environment variables will be `AZUREML_SWEEP_seed`,"
            " `AZUREML_SWEEP_model_unet`, `AZUREML_SWEEP_trainer_max_epochs`,"
            " and `AZUREML_SWEEP_model_learning_rate`."
        ),
        rich_help_panel=PANEL_COMMAND,
    ),
    sweep_prefix: Optional[str] = typer.Option(
        None,
        "--sweep-prefix",
        help=(
            "Prefix to prepend to the sweep arguments in the command. If not provided,"
            " the sweep arguments will not be added to the command."
        ),
        rich_help_panel=PANEL_COMMAND,
    ),
    sweep_max_concurrent_trials: Optional[int] = typer.Option(
        None,
        "--max-concurrent-trials",
        help=("Maximum number of concurrent trials for the sweep job."),
        rich_help_panel=PANEL_COMMAND,
    ),
    stream_logs: bool = typer.Option(
        False,
        "--stream-logs",
        "-l",
        help="Wait for completion and stream the logs of the job.",
    ),
    source_dir: Optional[Path] = typer.Option(
        None,
        "--source-dir",
        help=(
            "Path to the directory containing the source code for the job. If not "
            "provided, the current directory is used."
        ),
        rich_help_panel=PANEL_ENVIRONMENT,
    ),
    project_dir: Optional[Path] = typer.Option(
        None,
        "--project-dir",
        "-P",
        help=(
            "Directory containing a pyproject.toml, uv.lock and .python-version file."
            " These files will be used to build the Docker image."
            " If not provided, the current directory is used."
        ),
        rich_help_panel=PANEL_ENVIRONMENT,
    ),
    num_gpus: Optional[int] = typer.Option(
        None,
        help=(
            "Number of requested GPUs per node. This should typically match the number"
            " of GPUs in the compute target. If provided, the `PyTorchDistribution`"
            " will be selected. Otherwise, the `MpiDistribution` will be used and "
            " `--executable` should be set to `'torchrun --nproc-per-node auto'`"
            " for multi-GPU PyTorch runs. Must not be set for Lightning jobs."
            " More information at"
            " https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-distributed-gpu?view=azureml-api-2."
        ),
        rich_help_panel=PANEL_COMPUTE,
    ),
    debug: bool = typer.Option(
        False,
        help=(
            "Install debugpy on AML and run the command using debugpy. The job"
            " will not start until a remote debugger is attached. More information at"
            " https://learn.microsoft.com/en-us/azure/machine-learning/how-to-interactive-jobs?view=azureml-api-2&tabs=ui#attach-a-debugger-to-a-job."
        ),
        rich_help_panel=PANEL_INTERACTION,
    ),
    tensorboard: bool = typer.Option(
        get_default("enable_tensorboard"),
        help="Enable a TensorBoard interactive service for the job.",
        rich_help_panel=PANEL_INTERACTION,
    ),
    tensorboard_dir: Path = typer.Option(
        get_default("tensorboard_dir"),
        help="Directory in which the TensorBoard logs are expected to be stored.",
        rich_help_panel=PANEL_INTERACTION,
    ),
    profiler: bool = typer.Option(
        False,
        help="Enable profiling on Azure ML. Needs CUDA >= 12 and PyTorch >= 2.",
    ),
    dependency_groups: Optional[List[str]] = typer.Option(  # noqa: UP006, UP007
        None,
        "--dependency-group",
        "-G",
        help=(
            "Dependency groups to install in the Docker image."
            " If not provided, no dependency groups are installed."
            " The groups are defined in the pyproject.toml file."
            " This option can be used multiple times."
        ),
        rich_help_panel=PANEL_ENVIRONMENT,
    ),
    optional_dependencies: Optional[List[str]] = typer.Option(  # noqa: UP006, UP007
        None,
        "--extra",
        help=(
            "Optional dependency groups (extras) to install in the Docker image."
            " If not provided, no extras are installed."
            " The optional groups are defined in the pyproject.toml file."
            " This option can be used multiple times."
        ),
        rich_help_panel=PANEL_ENVIRONMENT,
    ),
    conda_env_file: Optional[Path] = typer.Option(
        None,
        "--conda-env-file",
        help=(
            "Path to a conda environment YAML file"
            " (e.g., environment.yml). If provided, a conda"
            " environment will be used instead of"
            " Docker build context."
            " Cannot be used together with --build-context, --aml-environment,"
            " or uv-specific options."
        ),
        rich_help_panel=PANEL_ENVIRONMENT,
        exists=True,
    ),
    only_environment: bool = typer.Option(
        False,
        "--only-env",
        help=(
            "Exit after instantiating the environment. This is useful during"
            " development so that the AML environment build runs immediately and the"
            " job starts faster once the script is ready to be submitted."
        ),
        rich_help_panel=PANEL_ENVIRONMENT,
    ),
    environment_variables: Optional[List[str]] = typer.Option(  # noqa: UP006, UP007
        None,
        "--set",
        "-E",
        help=(
            "Environment variables to set on the job."
            " The format is `KEY=VALUE`."
            " This option can be used multiple times."
        ),
        rich_help_panel=PANEL_ENVIRONMENT,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-D",
        help="Exit before submitting the job.",
    ),
    context: typer.Context = typer.Option(
        None,
        help="[Extra arguments to be added to the command]",
    ),
) -> None:
    r"""Submit a job to be run on Azure Machine Learning.

    Unrecognized arguments are ignored and propagated to the script.

    ```shell
    submit-aml \
        --script run.py \
        --experiment-name "my-experiment" \
        --mount "vindr_dir=VINDR-CXR-V2" \
            --my-script-arg "hello"
    ```
    """
    environment_variables_dict = get_env_variable_dict(environment_variables)
    sweep_inputs_dict = get_sweep_inputs_from_args(sweep_args)

    try:
        submit_to_aml(
            aml_environment=aml_environment,
            base_docker_image=docker_image,
            build_docker_context=build_docker_context,
            command_prefix=command_prefix,
            compute_target=compute_target,
            conda_env_file=conda_env_file,
            datasets_download=datasets_download,
            datasets_mount=datasets_mount,
            datasets_output=output,
            debug=debug,
            dependency_groups=dependency_groups,
            description=description,
            docker_run=docker_run,
            docker_shared_memory_gb=docker_shared_memory_gb,
            dry_run=dry_run,
            enable_profiler=profiler,
            enable_tensorboard=tensorboard,
            environment_variables=environment_variables_dict,
            executable=executable,
            experiment_name=experiment_name,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            only_environment=only_environment,
            optional_dependencies=optional_dependencies,
            project_dir=project_dir,
            resource_group=resource_group,
            run_name=run_name,
            script_args=context.args,
            script_path=script_path,
            source_dir=source_dir,
            subscription_id=subscription_id,
            sweep_inputs=sweep_inputs_dict,
            sweep_max_concurrent_trials=sweep_max_concurrent_trials,
            sweep_prefix=sweep_prefix,
            tensorboard_dir=tensorboard_dir,
            wait_for_completion=stream_logs,
            workspace_name=workspace_name,
        )
    except Exception:
        logger.critical("Failed to submit job to Azure ML. Reason:")
        console = Console()
        console.print_exception()


if __name__ == "__main__":
    app()
