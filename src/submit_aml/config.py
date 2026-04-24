"""Layered configuration system for ``submit-aml``.

Configuration is resolved with the following precedence (highest wins):

1. CLI flags (handled externally by the Typer layer)
2. Environment variables prefixed ``SUBMIT_AML_``
3. TOML config file at ``~/.config/submit-aml/config.toml``
4. Package defaults from :mod:`.defaults`
"""

from __future__ import annotations

import functools
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # ty: ignore[unresolved-import]

from .defaults import DEFAULT_COMMAND_PREFIX
from .defaults import DEFAULT_COMPUTE_TARGET
from .defaults import DEFAULT_DOCKER_IMAGE
from .defaults import DEFAULT_DOCKER_SHARED_MEMORY_GB
from .defaults import DEFAULT_ENABLE_TENSORBOARD
from .defaults import DEFAULT_EXECUTABLE
from .defaults import DEFAULT_NUM_NODES
from .defaults import DEFAULT_TENSORBOARD_DIR
from .logger import logger

_ENV_PREFIX: str = "SUBMIT_AML_"

_CONFIG_PATH_ENV_VAR: str = f"{_ENV_PREFIX}CONFIG"
"""Environment variable that overrides the TOML config file location."""


def _resolve_config_path() -> Path:
    """Resolve the TOML config path, honoring the override env var."""
    override = os.environ.get(_CONFIG_PATH_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path("~/.config/submit-aml/config.toml").expanduser()


CONFIG_PATH: Path = _resolve_config_path()
"""Default location for the user-level TOML config file.

Set the ``SUBMIT_AML_CONFIG`` environment variable to point at a different
file (or at a non-existent path to skip TOML config entirely).
"""

# ---------------------------------------------------------------------------
# Registry: flat key → (TOML (section, key), package default)
# ---------------------------------------------------------------------------

_CONFIG_KEYS: dict[str, tuple[tuple[str | None, str], Any]] = {
    "default_workspace": ((None, "default_workspace"), None),
    "compute_target": (("compute", "compute_target"), DEFAULT_COMPUTE_TARGET),
    "num_nodes": (("compute", "num_nodes"), DEFAULT_NUM_NODES),
    "docker_shared_memory_gb": (
        ("compute", "docker_shared_memory_gb"),
        DEFAULT_DOCKER_SHARED_MEMORY_GB,
    ),
    "docker_image": (("environment", "docker_image"), DEFAULT_DOCKER_IMAGE),
    "command_prefix": (("command", "command_prefix"), DEFAULT_COMMAND_PREFIX),
    "executable": (("command", "executable"), DEFAULT_EXECUTABLE),
    "tensorboard_dir": (
        ("tensorboard", "tensorboard_dir"),
        DEFAULT_TENSORBOARD_DIR,
    ),
    "enable_tensorboard": (
        ("tensorboard", "enable_tensorboard"),
        DEFAULT_ENABLE_TENSORBOARD,
    ),
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_toml(path: Path) -> dict[str, Any]:
    """Read and parse a TOML file.

    Args:
        path: Absolute path to the TOML file.

    Returns:
        Parsed TOML data as a nested dict.
    """
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _get_nested(data: dict[str, Any], section: str | None, key: str) -> Any | None:
    """Retrieve a value from a possibly nested dict.

    Args:
        data: Parsed TOML data.
        section: Top-level table name, or ``None`` for root-level keys.
        key: Key inside the table (or at the root when *section* is ``None``).

    Returns:
        The value if found, otherwise ``None``.
    """
    if section is None:
        return data.get(key)
    return data.get(section, {}).get(key)


def _coerce(value: str, reference: Any) -> Any:
    """Coerce a string (from an env var) to match *reference*'s type.

    Args:
        value: Raw string value from the environment.
        reference: The package default whose type guides coercion.

    Returns:
        The coerced value.
    """
    if reference is None:
        return value
    if isinstance(reference, bool):
        return value.lower() in {"1", "true", "yes"}
    if isinstance(reference, int):
        return int(value)
    if isinstance(reference, Path):
        return Path(value)
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@functools.cache
def get_config() -> dict[str, Any]:
    """Load and merge configuration from all non-CLI layers.

    Layers (lowest to highest priority):

    1. Package defaults from :mod:`.defaults`
    2. TOML config file (if present)
    3. Environment variables prefixed ``SUBMIT_AML_``

    The result is cached so the TOML file is read at most once per process.

    Returns:
        A flat dict mapping config key names to their resolved values.
    """
    config: dict[str, Any] = {}

    # Layer 1 – package defaults
    for key, (_, default) in _CONFIG_KEYS.items():
        config[key] = default

    # Layer 2 – TOML config file
    if CONFIG_PATH.is_file():
        logger.info("Loading config from {}", CONFIG_PATH)
        try:
            toml_data = _read_toml(CONFIG_PATH)
            for key, ((section, toml_key), _) in _CONFIG_KEYS.items():
                value = _get_nested(toml_data, section, toml_key)
                if value is not None:
                    config[key] = value
        except Exception as exc:
            logger.warning("Failed to read config file {}: {}", CONFIG_PATH, exc)
    else:
        logger.debug("No config file found at {}", CONFIG_PATH)

    # Layer 3 – environment variables
    for key, (_, default) in _CONFIG_KEYS.items():
        env_value = os.environ.get(f"{_ENV_PREFIX}{key.upper()}")
        if env_value is not None:
            config[key] = _coerce(env_value, default)

    return config


def get_default(key: str, fallback: Any = None) -> Any:
    """Get the resolved default for a single CLI option.

    This is the function the Typer layer should call to populate its
    ``default`` parameter so that TOML / env-var values are picked up
    automatically.

    Args:
        key: Flat config key (e.g. ``"subscription_id"``).
        fallback: Returned when the key is absent or ``None`` in every layer.

    Returns:
        The highest-priority non-``None`` value, or *fallback*.
    """
    value = get_config().get(key)
    return value if value is not None else fallback


def resolve_workspace_config(workspace_name: str) -> dict[str, str]:
    """Look up a workspace profile and return its settings.

    Reads the ``[workspaces.<workspace_name>]`` section from the TOML config
    file and returns any ``subscription_id`` and ``resource_group`` values
    defined there.

    Args:
        workspace_name: Name of the workspace profile to look up.

    Returns:
        A dict with ``subscription_id`` and/or ``resource_group`` keys if the
        profile exists and defines them, otherwise an empty dict.
    """
    if not CONFIG_PATH.is_file():
        return {}

    try:
        toml_data = _read_toml(CONFIG_PATH)
    except Exception as exc:
        logger.warning("Failed to read config file {}: {}", CONFIG_PATH, exc)
        return {}

    workspaces = toml_data.get("workspaces", {})
    profile = workspaces.get(workspace_name, {})

    if not profile:
        return {}

    result: dict[str, str] = {}
    for key in ("subscription_id", "resource_group"):
        value = profile.get(key)
        if value is not None:
            result[key] = value

    return result


def generate_template_config() -> str:
    """Return a TOML template that users can save to :data:`CONFIG_PATH`.

    Returns:
        A ready-to-write string with commented-out example values.
    """
    return textwrap.dedent("""\
        # submit-aml configuration file
        # Save to: ~/.config/submit-aml/config.toml

        # Default workspace used when --workspace is not passed.
        # default_workspace = "my-workspace"

        # ── Workspace profiles ──────────────────────────────────────────
        # Define named profiles so that --workspace <name> automatically
        # resolves subscription_id and resource_group.
        #
        # [workspaces.my-workspace]
        # subscription_id = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        # resource_group = "my-resource-group"
        #
        # [workspaces.other-workspace]
        # subscription_id = "yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy"
        # resource_group = "other-rg"

        [compute]
        # compute_target = "my-gpu-cluster"
        # num_nodes = 1
        # docker_shared_memory_gb = 256

        [environment]
        # docker_image = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04"

        [command]
        # command_prefix = "uv run --no-default-groups"
        # executable = "python"

        [tensorboard]
        # tensorboard_dir = "logs/tensorboard"
        # enable_tensorboard = true
    """)  # noqa: E501
