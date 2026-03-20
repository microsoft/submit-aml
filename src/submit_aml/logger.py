import logging

from loguru import logger
from rich.logging import RichHandler

from .defaults import DEFAULT_LOGGERS_TO_SUPPRESS

# https://github.com/Textualize/rich/issues/163#issuecomment-661023060
logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def suppress_azure_warnings(modules: list[str] | None = None) -> None:
    """Suppress noisy Azure SDK log messages by raising their log level to ERROR.

    Args:
        modules: List of logger module names to suppress. If ``None``, the
            default list from ``DEFAULT_LOGGERS_TO_SUPPRESS`` is used.
    """
    if modules is None:
        modules = DEFAULT_LOGGERS_TO_SUPPRESS
    for module in modules:
        logging.getLogger(module).setLevel(logging.ERROR)
