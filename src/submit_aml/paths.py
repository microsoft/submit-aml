from pathlib import Path


def get_cwd() -> Path:
    """Get the resolved current working directory."""
    return Path.cwd().resolve()
