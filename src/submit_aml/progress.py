import time
from contextlib import contextmanager

from loguru import logger
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn


class BarlessProgress(Progress):
    """A Rich progress display with a spinner and elapsed time, but no progress bar."""

    def __init__(self, *args, **kwargs):
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        ]
        super().__init__(*columns, *args, **kwargs)


@contextmanager
def report_time_fancy(start_msg: str, end_msg: str):
    """Context manager that shows a Rich progress bar and logs elapsed time.

    Displays a spinner with ``start_msg`` while the block executes, then logs
    ``end_msg`` together with the elapsed time on completion.

    Args:
        start_msg: Message shown in the progress bar during execution.
        end_msg: Message logged after the block completes.
    """
    begin = time.time()
    with BarlessProgress() as progress:
        task = progress.add_task(start_msg, total=1)
        yield
        progress.update(task, advance=1)
    end = time.time()
    delta = _natural_delta(end - begin)
    logger.success(f"{end_msg} in {delta}!")


@contextmanager
def report_time(start_msg: str, end_msg: str):
    """Context manager that logs start/end messages with elapsed time.

    Logs ``start_msg`` before the block executes and ``end_msg`` together with
    the elapsed time after it completes.

    Args:
        start_msg: Message logged before execution begins.
        end_msg: Message logged after the block completes.
    """
    begin = time.time()
    logger.info(start_msg)
    yield
    end = time.time()
    delta = _natural_delta(end - begin)
    logger.success(f"{end_msg} in {delta}!")


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
