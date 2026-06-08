import sys
import tempfile
import traceback
from datetime import datetime
from datetime import timezone
from pathlib import Path

from rich.panel import Panel
from rich.text import Text

from .logger import console
from .logger import logger


def _summarize_exception(exc: BaseException) -> str:
    """Return a concise one-line summary of an exception.

    The summary combines the exception class name with the first non-empty line
    of its message, so verbose multi-line errors are reduced to their headline.

    Args:
        exc: The exception to summarise.

    Returns:
        A single-line summary string.
    """
    message = str(exc).strip()
    if not message:
        return exc.__class__.__name__
    first_line = message.splitlines()[0].strip()
    return f"{exc.__class__.__name__}: {first_line}"


def write_traceback_log(exc: BaseException) -> Path:
    """Write the full traceback of an exception to a temporary log file.

    The log also records the command line and a timestamp to make the file
    useful for debugging and bug reports.

    Args:
        exc: The exception whose traceback should be saved.

    Returns:
        The path to the created log file.
    """
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="submit-aml-",
        suffix=".log",
        delete=False,
        encoding="utf-8",
    )
    with handle:
        handle.write(f"Command: {' '.join(sys.argv)}\n")
        handle.write(f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n\n")
        handle.write(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        )
    return Path(handle.name)


def report_exception(exc: BaseException, *, message: str) -> Path:
    """Report an exception to the user without dumping the full traceback.

    A concise, pretty error panel is printed together with the path to a
    temporary log file containing the complete traceback.

    Args:
        exc: The exception to report.
        message: A short, human-readable description of what failed.

    Returns:
        The path to the log file holding the full traceback.
    """
    log_path = write_traceback_log(exc)

    body = Text()
    body.append(message, style="bold")
    body.append("\n\n")
    body.append(_summarize_exception(exc), style="red")

    console.print()
    console.print(
        Panel(
            body,
            title="Error",
            title_align="left",
            border_style="red",
        )
    )
    logger.info(f"Full traceback written to {log_path}")
    return log_path
