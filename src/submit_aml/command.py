import re
import sys
from pathlib import Path

from azure.ai.ml.entities._job.job_service import JobService
from azure.ai.ml.entities._job.job_service import JupyterLabJobService
from azure.ai.ml.entities._job.job_service import SshJobService
from azure.ai.ml.entities._job.job_service import TensorBoardJobService
from azure.ai.ml.entities._job.job_service import VsCodeJobService
from azure.ai.ml.entities._job.sweep.search_space import SweepDistribution
from azure.ai.ml.sweep import Choice

from .data import TypeInputsDict
from .logger import logger

_TypeService = (
    JobService | JupyterLabJobService | SshJobService | TensorBoardJobService | VsCodeJobService
)
TypeServices = dict[str, _TypeService]
TypeSweepValues = list[int | float | str | dict]


def build_command(
    command_prefix: str,
    executable: str,
    script_path: str,
) -> str:
    """Builds a command string for executing a script with the given executable.

    Examples:
        >>> build_command("uv run", "python", "script.py")
        'uv run python script.py'
        >>> build_command("", "nvidia-smi", "")
        'nvidia-smi'
        >>> build_command("uv run --with pyright", "pyright")
        'uv run --with pyright pyright'

    Args:
        command_prefix: The prefix to prepend to the command (can be empty).
        executable: The executable to run the script with, if a script is passed.
        script_path: The path to the script to execute (can be empty).
    """
    command = executable
    if command_prefix:
        command = command_prefix + " " + command
    if script_path:
        command += " " + script_path
    return command


def log_command(command: str, script_args: list[str]) -> None:
    """Log the command and its script arguments.

    Args:
        command: The command string to log.
        script_args: List of arguments passed to the script.
    """
    logger.info("Command:")
    logger.info(f'  "{command}"')
    if script_args:
        logger.info("Arguments:")
        for arg in script_args:
            logger.info(f"  {arg}")


def build_debug_command(
    command_prefix: str,
    executable: str,
    port: int = 5678,
) -> tuple[str, str]:
    """Build a command prefix and executable for debugging with debugpy.

    Only supports "python" as the executable.

    If `"uv run"` is in the command prefix, `--with debugpy` will be added to it so that
    `debugpy` is added to the ephemeral environment.

    Args:
        command_prefix: The prefix to prepend to the command (can be empty).
        executable: The executable to run the script with.
        port: The port for debugpy to listen on.

    Returns:
        A tuple containing the modified command prefix and executable.

    Examples:
        >>> build_debug_command("uv run", "python")
        ('uv run --with debugpy', 'python -m debugpy --listen localhost:5678 --wait-for-client')
        >>> build_debug_command("", "python")
        ('pip install debugpy &&', 'python -m debugpy --listen localhost:5678 --wait-for-client')
    """
    if command_prefix.startswith("uv run"):
        logger.info('Adding debugpy to uv run command using "--with"')
        command_prefix += " --with debugpy"
    else:
        pip_install_string = "pip install debugpy"
        logger.info(f'Adding "{pip_install_string}" to command prefix')
        if command_prefix:
            command_prefix = f"{command_prefix} && {pip_install_string} &&"
        else:
            command_prefix = f"{pip_install_string} &&"
    options = f"--listen localhost:{port} --wait-for-client"
    if executable == "python":
        executable += f" -m debugpy {options}"
    else:
        msg = (
            'Debugging is only supported when using "python" as the executable.'
            f' Got "{executable}".'
        )
        logger.error(msg)
        sys.exit(1)
    return command_prefix, executable


def add_service_for_debugging(services: TypeServices) -> None:
    """Add a VS Code service for debugging to the provided services dictionary.

    Args:
        services: Dictionary of job services to which the debugging service
            will be added.
    """
    services["debugging"] = VsCodeJobService()


def add_service_for_tensorboard(services: TypeServices, tensorboard_dir: Path) -> None:
    """Add a TensorBoard service to the provided services dictionary.

    If a TensorBoard service already exists, a warning is logged and no new
    service is added.

    Args:
        services: Dictionary of job services to which the TensorBoard service
            will be added.
        tensorboard_dir: Directory where TensorBoard log files are stored.
    """
    for service in services.values():
        if isinstance(service, TensorBoardJobService):
            msg = "TensorBoard service already exists. Not adding another one."
            logger.warning(msg)
            break
    else:
        services["tensorboard"] = TensorBoardJobService(log_dir=str(tensorboard_dir))


def sanitize_input_name(input_name: str) -> str:
    """Replace with underscores some characters that are not letters and underscores."""
    sanitized_name = input_name.strip("+").replace("/", "_").replace(".", "_")
    return sanitized_name


def _parse_value_string(value_str: str) -> int | float | str:
    """Parse a string representation of a value into an integer, float, or string."""
    value_str = value_str.strip()
    value: int | float | str
    # integer
    if re.match(r"^[\d]+$", value_str):
        value = int(value_str)
    # float with possible scientific notation
    elif re.match(r"^[\d.]+(e[\+-]?[\d]+)?$", value_str):
        value = float(value_str)
    # string (single quoted)
    elif re.match(r"^'.+'$", value_str):
        value = value_str[1:-1]
    # string (double quoted)
    elif re.match(r'^".+"$', value_str):
        value = value_str[1:-1]
    # string (unquoted)
    elif re.match(r"^[\w-]+$", value_str):
        value = value_str
    else:
        msg = f"Cannot convert string: {value_str}"
        raise ValueError(msg)
    return value


def _parse_values_string(values_str: str) -> list[int | float | str]:
    """Parse a comma-separated string of values into a list of integers, floats, or strings."""
    values = []
    for value_str in values_str.split(","):
        value = _parse_value_string(value_str.strip())
        values.append(value)
    return values


def _parse_sweep_arg(sweep_arg: str) -> tuple[str, str, str, list[int | float | str]]:
    r"""Parse arguments for AML hyperparameter sweep.

    Parses a string of the form "parameter=[value1, value2, ...]" into a tuple containing:
    - The sanitized parameter name (replacing slashes and dots with underscores) for AML.
    - The original parameter path (with slashes and dots).
    - A list of values parsed from the string.

    Examples:
    - "seed=[0, 1, 2]" -> ("seed", "", "seed", [0, 1, 2])
    - "model/unet=[\'tiny\', \'small\']" -> ("model_unet", "", "model/unet", ["tiny", "small"])
    - "+trainer.max_epochs=[10, 20]" -> ("trainer_max_epochs", "+", "trainer.max_epochs", [10, 20])
    - "model.learning_rate=[1.0e-2, 2.0e-2]"
      -> ("model_learning_rate", "", "model.learning_rate", [0.01, 0.02])
    """
    match = re.match(r"(\+{0,2}?)([\w./]+)=\[(.+)\]", sweep_arg)
    if not match:
        raise ValueError(f"Invalid sweep argument: {sweep_arg}")

    prefix: str = match.group(1)
    raw_name: str = match.group(2)
    values_str: str = match.group(3)
    sanitized_name = sanitize_input_name(raw_name)
    values = _parse_values_string(values_str)

    return sanitized_name, prefix, raw_name, values


def get_sweep_inputs_from_args(
    sweep_args: list[str] | None,
    distribution_class: type[Choice] = Choice,
) -> dict[str, SweepDistribution]:
    """Parse sweep arguments and return a dictionary of sweep inputs.

    The input names must be sanitized before passing them to the function to instantiate
    the command.
    """
    if sweep_args is None:
        return {}
    sweep_inputs = {}
    for arg in sweep_args:
        _, prefix, raw_name, values = _parse_sweep_arg(arg)
        distribution = distribution_class(values=list(values))
        sweep_inputs[prefix + raw_name] = distribution

    return sweep_inputs


def add_sweep_to_inputs_and_script_args(
    sweep_inputs: dict[str, SweepDistribution],
    inputs: TypeInputsDict,
    script_args: list[str],
    sweep_prefix: str | None = None,
) -> tuple[TypeInputsDict, list[str]]:
    """Add sweep inputs to the inputs dictionary and (if needed) command string.

    Args:
        sweep_inputs: A dictionary of sweep inputs where keys are raw parameter names
            and values are `SweepDistribution` objects.
        inputs: A dictionary to which the sweep inputs will be added.
        script_args: Current arguments to submitted script. The sweep inputs will be
            appended to this list if `sweep_prefix` is provided.
        sweep_prefix: An optional prefix to prepend to each parameter in the command.
    """

    for raw_name, argument in sweep_inputs.items():
        sanitized_name = sanitize_input_name(raw_name)
        if raw_name != sanitized_name:
            msg = f'Sanitized sweep input name: "{raw_name}" to "{sanitized_name}"'
            logger.warning(msg)
        inputs[sanitized_name] = argument
        if sweep_prefix is not None:
            script_args.append(f" {sweep_prefix}{sanitized_name}")
            script_args.append(" ${{inputs." + sanitized_name + "}}")
    return inputs, script_args
