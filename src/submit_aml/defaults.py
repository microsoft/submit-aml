"""Default configuration values for Azure ML job submission."""

from pathlib import Path

# --- Azure ML workspace (must be provided by the user) -------------------

DEFAULT_COMPUTE_TARGET: str | None = None

# --- uv commands ----------------------------------------------------------

DEFAULT_COMMAND_PREFIX: str = " ".join(["uv run", "--no-default-groups"])
DEFAULT_UV_SYNC_COMMAND: str = " ".join(
    [
        "uv sync",
        "--verbose",
        "--no-install-local",
        "--no-default-groups",
    ]
)

# --- Compute & Docker -----------------------------------------------------

DEFAULT_EXECUTABLE: str = "python"
DEFAULT_NUM_NODES: int = 1
# See more options at https://github.com/Azure/AzureML-Containers/tree/master/images
DEFAULT_OPENMPI_IMAGE: str = "openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04"
DEFAULT_DOCKER_IMAGE: str = f"mcr.microsoft.com/azureml/{DEFAULT_OPENMPI_IMAGE}"
DEFAULT_DOCKER_SHARED_MEMORY_GB: int = 256

# --- Sweep & TensorBoard --------------------------------------------------

DEFAULT_SWEEP_ALGORITHM: str = "grid"
DEFAULT_TENSORBOARD_DIR: Path = Path("logs/tensorboard")
DEFAULT_ENABLE_TENSORBOARD: bool = True

# --- Logging ---------------------------------------------------------------

DEFAULT_LOGGERS_TO_SUPPRESS: list[str] = [
    # pathOnCompute is not a known attribute of class <class 'azure.ai.ml. [...]
    "msrest.serialization",  # happens when using outputs
    # [...] Class X: This is an experimental class, and may change at any time. [...]
    "azure.ai.ml._utils._experimental",
]
