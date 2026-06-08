import time
from collections.abc import Iterator
from contextlib import contextmanager

from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn

from .logger import _INDENT
from .logger import console
from .logger import get_depth
from .logger import indent
from .logger import logger

# Whether a spinner is currently being displayed. Rich allows only one live
# display per console at a time, so nested spinners fall back to plain logging.
_spinner_active = False


class BarlessProgress(Progress):
    """A Rich progress display with a spinner and elapsed time, but no progress bar."""

    def __init__(self, *args, **kwargs):
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        ]
        super().__init__(*columns, *args, console=console, **kwargs)


@contextmanager
def report_time(
    start_msg: str,
    end_msg: str,
    *,
    spinner: bool = True,
) -> Iterator[None]:
    """Show a spinner while a block runs and report the elapsed time.

    While the block executes, a spinner with a live elapsed-time counter is
    displayed next to ``start_msg`` so the user gets feedback that work is in
    progress. Output emitted inside the block is indented one level and rendered
    above the spinner. When the block completes, the spinner is cleared and
    ``end_msg`` is logged together with the elapsed time.

    The spinner is skipped (``start_msg`` is logged as a plain header instead)
    when ``spinner`` is ``False``, when the console is not a terminal, or when a
    spinner is already active (rich allows only one live display per console).
    Disable the spinner for operations that render their own progress output
    (e.g. uploads), so the two live displays do not clash.

    Args:
        start_msg: Message shown next to the spinner during execution.
        end_msg: Message logged after the block completes.
        spinner: Whether to display a spinner. Set to ``False`` for operations
            that print their own progress.

    Yields:
        ``None``.
    """
    global _spinner_active
    begin = time.time()

    if not spinner or _spinner_active or not console.is_terminal:
        logger.info(start_msg)
        with indent():
            yield
    else:
        description = f"{_INDENT * get_depth()}{start_msg}"
        _spinner_active = True
        try:
            with BarlessProgress(transient=True) as progress:
                progress.add_task(description, total=None)
                with indent():
                    yield
        finally:
            _spinner_active = False

    delta = _natural_delta(time.time() - begin)
    logger.success(f"{end_msg} in {delta}.")


def _natural_delta(delta_seconds: float) -> str:
    """Return a human-readable string representing the time delta.

    We assume hours are never needed.

    Examples:
        >>> _natural_delta(1)
        '1 second'
        >>> _natural_delta(2)
        '2 seconds'
        >>> _natural_delta(60)
        '1 minute'
        >>> _natural_delta(61)
        '1 minute and 1 second'
        >>> _natural_delta(65)
        '1 minute and 5 seconds'
        >>> _natural_delta(120)
        '2 minutes'
        >>> _natural_delta(121)
        '2 minutes and 1 second'
        >>> _natural_delta(125)
        '2 minutes and 5 seconds'
    """
    minutes, seconds = divmod(delta_seconds, 60)
    if minutes == 0 and seconds < 1:
        return "less than a second"
    minutes = int(round(minutes))
    seconds = int(round(seconds))
    seconds_string = f"{seconds} second{'s' if seconds != 1 else ''}"
    if minutes < 1:
        return seconds_string
    minutes_string = f"{minutes} minute{'s' if minutes != 1 else ''}"
    if seconds < 1:
        return minutes_string
    return f"{minutes_string} and {seconds_string}" if minutes > 0 else seconds_string
