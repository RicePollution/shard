"""Live status feed for long-running CLI operations."""

from __future__ import annotations

import shutil
import sys
import threading

_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class StatusFeed:
    """Context manager that shows a single rewriting status line on stderr.

    Uses ``\\r`` and ANSI escape ``\\033[K`` (erase to end of line) to
    rewrite the same terminal line in place.  Truncates text to terminal
    width so it never wraps.  Becomes a silent no-op when stderr is not
    a TTY (CI, pipes, redirected output).

    Usage::

        with StatusFeed() as status:
            status.update("Extracting content from URL...")
            content = extractor.run(source)
            status.update("Formatting notes...")
            result = formatter.format(content)
        # line clears automatically, normal output resumes
    """

    def __init__(self) -> None:
        self._is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        self._spinner_index = 0
        self._lock = threading.Lock()
        self._active = False

    def __enter__(self) -> StatusFeed:
        self._active = True
        return self

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self._active = False
        self.clear()

    def update(self, message: str) -> None:
        """Rewrite the status line with *message* and an animated spinner."""
        if not self._is_tty:
            return
        with self._lock:
            if not self._active:
                return
            spinner = _SPINNER_CHARS[self._spinner_index % len(_SPINNER_CHARS)]
            self._spinner_index += 1
            width = shutil.get_terminal_size().columns
            line = f"{spinner} {message}"
            if len(line) > width - 2:
                line = line[: width - 2]
            sys.stderr.write(f"\r\033[K{line}")
            sys.stderr.flush()

    def clear(self) -> None:
        """Clear the status line completely."""
        if not self._is_tty:
            return
        with self._lock:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
