import contextvars
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.text import Text

from .defaults import DEFAULT_LOGGERS_TO_SUPPRESS

console = Console()
"""Shared rich console used for all console output.

Reusing a single console instance lets log lines render cleanly above an active
spinner (rich requires the same console for both the live display and any other
output).
"""

_highlighter = ReprHighlighter()
"""Rich highlighter that colours numbers, paths, quoted strings, URLs, etc."""

_INDENT = "  "
"""String used for a single level of indentation."""

_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "submit_aml_log_depth",
    default=0,
)
"""Current logging depth, used to indent nested output."""

# Glyph and rich style for each log level. The glyphs are flat (non-emoji)
# symbols so they share a consistent look and only the colour varies.
_LEVEL_STYLES: dict[str, tuple[str, str]] = {
    "DEBUG": ("·", "dim"),
    "INFO": ("•", "cyan"),
    "SUCCESS": ("✓", "green"),
    "WARNING": ("▲", "yellow"),
    "ERROR": ("✗", "red"),
    "CRITICAL": ("✗", "bold red"),
}

# Levels for which the whole message (not just the glyph) is coloured.
_COLOURED_MESSAGE_LEVELS = frozenset({"DEBUG", "WARNING", "ERROR", "CRITICAL"})


@contextmanager
def indent() -> Iterator[None]:
    """Increase the logging depth within the context.

    Any output emitted through [`logger`][submit_aml.logger.logger] (or a
    spinner) while this context is active is indented one extra level. The
    previous depth is restored on exit, even if an exception is raised.

    Yields:
        ``None``.
    """
    token = _depth.set(_depth.get() + 1)
    try:
        yield
    finally:
        _depth.reset(token)


def get_depth() -> int:
    """Return the current logging depth.

    Returns:
        The number of active indentation levels.
    """
    return _depth.get()


def format_log_line(
    level_name: str,
    message: str,
    depth: int,
    *,
    highlight: bool = True,
    width: int | None = None,
) -> Text:
    """Render a log message as indented, glyph-prefixed rich text.

    The message is prefixed with ``depth`` levels of indentation and a coloured
    glyph for the level. Long or multi-line messages are wrapped with a hanging
    indent so continuation lines align under the message. For informational
    levels the message is passed through rich's highlighter so numbers, paths,
    quoted strings and URLs are colourised.

    Args:
        level_name: Name of the log level (e.g. ``"INFO"``).
        message: The (already interpolated) message to render.
        depth: Number of indentation levels to apply.
        highlight: Whether to apply rich highlighting to the message. Ignored
            for levels whose whole message is already coloured.
        width: Total width available for the rendered line. Defaults to the
            shared console width.

    Returns:
        A [`rich.text.Text`][] instance ready to be printed.
    """
    glyph, style = _LEVEL_STYLES.get(level_name, ("•", ""))
    text_style = style if level_name in _COLOURED_MESSAGE_LEVELS else ""

    indentation = _INDENT * depth
    prefix_width = len(indentation) + len(glyph) + 1
    continuation = " " * prefix_width

    message_text = Text(message, style=text_style)
    if highlight and not text_style:
        _highlighter.highlight(message_text)

    if width is None:
        width = console.size.width
    wrap_width = max(width - prefix_width, 1)
    lines = message_text.wrap(console, wrap_width)

    text = Text()
    for index, line in enumerate(lines):
        if index > 0:
            text.append("\n")
        if index == 0:
            text.append(indentation)
            text.append(glyph, style=style)
            text.append(" ")
        else:
            text.append(continuation)
        text.append_text(line)
    return text


class Logger:
    """Minimal rich-backed logger with a loguru-compatible interface.

    Each level method accepts an optional set of positional and keyword
    arguments. When any are given, the message is formatted with
    ``message.format(*args, **kwargs)``; otherwise it is used verbatim, so
    f-string messages containing literal braces are left untouched.
    """

    def _log(
        self,
        level_name: str,
        message: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        console.print(format_log_line(level_name, message, _depth.get()))

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self._log("DEBUG", message, args, kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an informational message."""
        self._log("INFO", message, args, kwargs)

    def success(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a success message."""
        self._log("SUCCESS", message, args, kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self._log("WARNING", message, args, kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self._log("ERROR", message, args, kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message."""
        self._log("CRITICAL", message, args, kwargs)


logger = Logger()
"""Module-level logger singleton used throughout the package."""


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
