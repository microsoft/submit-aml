import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from azure.ai.ml import MLClient
from azure.ai.ml.entities import BuildContext
from azure.ai.ml.entities import Environment
from azure.core.exceptions import ResourceNotFoundError

from .defaults import DEFAULT_DOCKER_IMAGE
from .defaults import DEFAULT_UV_SYNC_COMMAND
from .logger import logger


def get_env_variable_dict(
    environment_variables_list: list[str] | None,
) -> dict[str, str]:
    """Convert a list of `"KEY=VALUE"` environment variables to a dictionary.

    Args:
        environment_variables_list: Environment variables in the format `"KEY=VALUE"`.

    Returns:
        Dictionary with environment variable names as keys and their values as values.
    """
    if environment_variables_list is None:
        return {}
    environment_variables_dict = {}
    for item in environment_variables_list:
        # Check formatting
        if item.count("=") != 1:
            msg = (
                f'Invalid environment variable format: "{item}", expected "KEY=VALUE".'
            )
            raise ValueError(msg)
        key, value = item.split("=")
        environment_variables_dict[key] = value
    return environment_variables_dict


def log_environment_variables(environment: dict[str, Any]) -> None:
    """Log the environment variables, nicely aligned.

    Does nothing if the environment is empty.

    Args:
        environment: Dictionary of environment variables.

    Examples:
        >>> log_environment_variables({"VAR1": "value1", "VARIABLE2": "value2"})
        Environment variables:
          VAR1     : value1
          VARIABLE2: value2
    """
    if not environment:
        return
    logger.info("Environment variables:")
    max_key_len = max(len(key) for key in environment)
    for key in sorted(environment):
        logger.info(f'  {key:<{max_key_len}}: "{environment[key]}"')


def add_profiler_env_variables(environment: dict[str, str]) -> None:
    """Add environment variables to enable profiler on AML.

    These values were shared with Fernando by Chakrapani Ravi Kiran S on 4 April 2025.

    Args:
        environment: Dictionary of environment variables to update with the
            profiler configuration.
    """
    profiler_config = {
        "ENABLE_AZUREML_TRAINING_PROFILER": "true",
        "AZUREML_PROFILER_WAIT_DURATION_SECOND": "10",
        "AZUREML_PROFILER_RUN_DURATION_MILLISECOND": "10000",
        "KINETO_DAEMON_INIT_DELAY_S": "3",
    }
    environment.update(profiler_config)


def _check_lock_file_up_to_date(project_dir: Path) -> None:
    """Check if the `uv.lock` file is up to date with the `pyproject.toml`.

    If the lock file is not up to date, log an error message and exit.

    Args:
        project_dir: Path to the project directory containing the `pyproject.toml`
            and `uv.lock` files.
    """
    command = f"uv lock --check --project {project_dir}"
    try:
        subprocess.run(
            command.split(),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = (
            'The "uv.lock" file is not up to date with the "pyproject.toml" file.'
            f' Run "uv lock --project {project_dir}" to update it.'
        )
        raise RuntimeError(msg) from e


def _check_env_files(project_dir: Path) -> tuple[Path, Path, Path]:
    """Check if the required environment files exist in the project directory.

    Args:
        project_dir: Path to the project directory containing the `pyproject.toml`,
            `uv.lock`, and `.python-version` files.
    """
    pyproject_path = project_dir / "pyproject.toml"
    if not pyproject_path.exists():
        msg = (
            'A "pyproject.toml" file is required to build the environment.'
            f' Run "uv init --project {project_dir}" to create it.'
        )
        raise FileNotFoundError(msg)

    uv_lock_path = project_dir / "uv.lock"
    if not uv_lock_path.exists():
        msg = (
            'A "uv.lock" file is required to build a reproducible environment.'
            f' Run "uv lock --project {project_dir}" to create it.'
        )
        raise FileNotFoundError(msg)

    pinned_python_path = project_dir / ".python-version"
    if not pinned_python_path.exists():
        msg = (
            'A ".python-version" file is required to build a reproducible environment.'
            f' Run "uv python pin <version> --project {project_dir}" to create it.'
        )
        raise FileNotFoundError(msg)

    return pyproject_path, uv_lock_path, pinned_python_path


def generate_build_context(
    project_dir: Path,
    base_docker_image: str = DEFAULT_DOCKER_IMAGE,
    docker_run: str | None = None,
    uv_sync_command: str = DEFAULT_UV_SYNC_COMMAND,
    dependency_groups: list[str] | None = None,
    optional_dependencies: list[str] | None = None,
) -> BuildContext:
    """Instantiate an AML build context for the job.

    The generated Dockerfile installs uv, creates the virtual environment, and installs
    the (remote) with uv sync, using the `pyproject.toml`, `uv.lock` and
    `.python-version` files.

    The files are copied to a temporary directory which is used as the build context.

    By default, Python files are compiled to bytecode (`__pycache__/*.pyc`) after
    installation. This option is enabled to trade longer installation times for faster
    start times.

    Args:
        project_dir: Path to the project directory containing the `pyproject.toml`,
            `uv.lock`, and `.python-version` files.
        base_docker_image: Base Docker image from which the AML environment is built.
        docker_run: Additional command to run in the Dockerfile at the beginning of
            the build process. Useful for e.g. `apt-get` packages.
        uv_sync_command: Base uv sync command to create the virtual environment.
        dependency_groups: Optional list of dependency groups from `pyproject.toml` to
            include in the `uv sync` command.
    """
    pyproject_path, uv_lock_path, pinned_python_path = _check_env_files(project_dir)
    _check_has_patch(pinned_python_path)
    _check_lock_file_up_to_date(project_dir)

    prefix = f"{__package__}-{project_dir.name}-"
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    logger.info(f"Temporary directory for build context: {temp_dir}")
    shutil.copy(pyproject_path, temp_dir)
    shutil.copy(uv_lock_path, temp_dir)
    shutil.copy(pinned_python_path, temp_dir)

    # Add dependency groups and optional dependencies to the uv sync command
    if dependency_groups is not None:
        for group in dependency_groups:
            uv_sync_command += f" --group {group}"
    if optional_dependencies is not None:
        for dependency in optional_dependencies:
            uv_sync_command += f" --extra {dependency}"

    dockerfile_template_path = Path(__file__).parent / "Dockerfile"
    if docker_run is None:
        docker_run = ""
    else:
        docker_run = f"\nRUN {docker_run}\n"
    dockerfile_string = dockerfile_template_path.read_text().format(
        base_docker_image=base_docker_image,
        uv_sync_command=uv_sync_command,
        docker_run=docker_run,
    )
    logger.info("Sync command in Docker build:")
    logger.info(f'  "{uv_sync_command}"')
    dockerfile_path = temp_dir / "Dockerfile"
    dockerfile_path.write_text(dockerfile_string)

    build_context = BuildContext(path=str(temp_dir))
    return build_context


def _check_has_patch(python_version_path: Path) -> None:
    """Log a warning if the .python-version file does not contain a patch version.

    The file should contain a version like "3.12.10", not just "3.12".

    Args:
        python_version_path: Path to the .python-version file.
    """
    python_version = python_version_path.read_text().strip()
    if len(python_version.split(".")) < 3:
        msg = (
            f'The .python-version file specified Python version "{python_version}",'
            " but should contain a full Python version, including"
            " the patch number (e.g., 3.11.10)."
            " Run `uv python pin <version_with_patch>`"
            " in the project directory to pin it."
        )
        logger.warning(msg)


def generate_conda_environment(
    conda_env_file: Path,
    base_docker_image: str,
) -> Environment:
    """Create an AML Environment from a conda environment YAML file.

    Args:
        conda_env_file: Path to the conda environment YAML file (e.g., conda.yaml).
        base_docker_image: Base Docker image from which the AML environment is built.

    Returns:
        An AML Environment instance configured with the conda environment.
    """
    if not conda_env_file.exists():
        raise FileNotFoundError(f"Conda environment file not found: {conda_env_file}")

    conda_content = conda_env_file.read_text()
    content_hash = hashlib.md5(conda_content.encode("utf-8")).hexdigest()
    environment_name = f"conda-env-{content_hash}"

    logger.info(f'Creating AML Environment "{environment_name}" from conda file.')
    environment = Environment(
        name=environment_name,
        conda_file=str(conda_env_file),
        image=base_docker_image,
    )
    return environment


def infer_environment(
    ml_client: MLClient | None = None,
    project_dir: Path | None = None,
    base_docker_image: str | None = None,
    dependency_groups: list[str] | None = None,
    optional_dependencies: list[str] | None = None,
    aml_environment: str | None = None,
    conda_env_file: Path | None = None,
    docker_run: str | None = None,
    *,
    build_docker_context: bool = True,
    dry_run: bool = False,
) -> str | Environment | None:
    """Infer the type and characteristics of the environment to use for the job.

    Args:
        ml_client: The MLClient instance to use for registering the environment or
            checking for an existing one. May be `None` if `aml_environment` is `None`
            and `build_docker_context` is `False`.
        project_dir: Path to the project directory containing the `pyproject.toml`,
            `uv.lock`, and `.python-version` files. May be `None` if
            `build_docker_context` is `False`.
        base_docker_image: Base Docker image from which the AML environment is built.
            May be `None` if `aml_environment` is provided.
        dependency_groups: Dependency groups from `pyproject.toml` to include in the
            `uv sync` command when building the Docker context.
        optional_dependencies: Optional dependencies (extras) to include in the
            `uv sync` command when building the Docker context.
        aml_environment: Name of an existing Azure ML environment to use. If provided,
            the latest version of the environment will be used.
        conda_env_file: Path to a conda environment YAML file to create the environment
            from. If provided, this takes precedence over building a Docker context.
        docker_run: Additional command to run in the Dockerfile at the beginning of
            the build process. Useful for e.g. `apt-get` packages.
        build_docker_context: Whether to build a Docker context for the environment.
        dry_run: If `True`, the function will not register the environment but will
            still return the name of the environment that would be registered. This has
            no effect if `aml_environment` is provided or `build_docker_context` is
            `False`.

    Returns:
        The name of the environment to use for the job, or an instance of `Environment`
        if only a Docker image is specified.
    """
    if aml_environment is not None:
        logger.info(f'Using existing AML environment: "{aml_environment}"')
        if ml_client is None:
            msg = (
                "An instance of MLClient must be provided to check the version of the"
                f' environment "{aml_environment}".'
            )
            raise ValueError(msg)
        latest_environment = max(
            ml_client.environments.list(name=aml_environment),
            key=lambda e: (
                int(e.version) if e.version is not None and e.version.isdigit() else -1
            ),
        )
        assert latest_environment.version is not None, (
            f'Could not find a valid version for environment "{aml_environment}"'
        )
        environment = f"azureml:{aml_environment}:{latest_environment.version}"
    elif conda_env_file is not None:
        assert base_docker_image is not None, (
            "base_docker_image cannot be None when using a conda_env_file"
        )
        environment_instance = generate_conda_environment(
            conda_env_file, base_docker_image=base_docker_image
        )
        if dry_run:
            logger.info(
                "Dry run mode enabled. Will not try to register the conda environment."
            )
            return environment_instance.name
        if ml_client is None:
            msg = (
                "An instance of MLClient must be provided"
                " to register the conda environment."
            )
            raise ValueError(msg)
        registered_environment = _register_environment(ml_client, environment_instance)
        assert registered_environment.name is not None
        environment = f"{registered_environment.name}:{registered_environment.version}"
    elif build_docker_context:
        msg_suffix = "must be provided to generate the build context."
        if project_dir is None:
            raise ValueError(f"Project directory {msg_suffix}")
        if base_docker_image is None:
            raise ValueError(f"Base Docker image {msg_suffix}")
        build_context = generate_build_context(
            project_dir,
            base_docker_image=base_docker_image,
            docker_run=docker_run,
            dependency_groups=dependency_groups,
            optional_dependencies=optional_dependencies,
        )
        environment_instance = Environment(build=build_context)
        if dry_run:
            logger.info(
                "Dry run mode enabled. Will not try to register the environment."
            )
            return environment_instance.name
        if ml_client is None:
            msg = (
                "An instance of MLClient must be provided to register the environment."
            )
            raise ValueError(msg)
        registered_environment = _register_environment(ml_client, environment_instance)
        assert registered_environment.name is not None
        environment = f"{registered_environment.name}:{registered_environment.version}"
    elif base_docker_image is not None:
        environment = Environment(image=base_docker_image)
    else:
        raise ValueError(
            "Either `build_docker_context` must be True,"
            " or `aml_environment` must be"
            " provided, or `conda_env_file` must be"
            " provided, or `base_docker_image`"
            " must be provided."
        )
    return environment


def _register_environment(ml_client: MLClient, environment: Environment) -> Environment:
    """Register the environment with Azure ML if it does not already exist.

    Args:
        environment: Instance of `Environment` to be registered.
        ml_client: Instance of `MLClient` to use for registering the environment.

    Returns:
        The registered environment.
    """
    try:
        assert environment.name is not None
        if environment.version is None:
            kwargs = {"label": "latest"}
        else:
            kwargs = {"version": environment.version}
        logger.info(
            f'Checking if environment "{environment.name}" ({kwargs}) exists...'
        )
        env = ml_client.environments.get(environment.name, **kwargs)
        msg = (
            f'Found a registered environment with name "{environment.name}"'
            f' and version "{env.version}"'
        )
        logger.info(msg)
    except ResourceNotFoundError:
        logger.info("Environment not found. Registering a new one...")
        env = ml_client.environments.create_or_update(environment)
        logger.info(f'Registered environment: "{env.name}" (version: "{env.version}")')
    return env
