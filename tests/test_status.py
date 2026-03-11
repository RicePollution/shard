"""Tests for the live status feed."""

from __future__ import annotations

import io
import os
from unittest.mock import patch

from shard.ui.status import StatusFeed, _SPINNER_CHARS


class TestStatusFeedUpdate:
    """Verify update() rewrites the same line."""

    def test_update_writes_carriage_return(self) -> None:
        buf = io.StringIO()
        buf.isatty = lambda: True  # type: ignore[assignment]

        with patch("shard.ui.status.sys.stderr", buf):
            with StatusFeed() as status:
                status.update("Extracting...")

        output = buf.getvalue()
        assert "\r" in output
        assert "Extracting..." in output

    def test_update_includes_spinner_character(self) -> None:
        buf = io.StringIO()
        buf.isatty = lambda: True  # type: ignore[assignment]

        with patch("shard.ui.status.sys.stderr", buf):
            with StatusFeed() as status:
                status.update("Working...")

        output = buf.getvalue()
        assert output.lstrip("\r\033[K")[0] in _SPINNER_CHARS


class TestStatusFeedClear:
    """Verify clear() removes the line."""

    def test_clear_erases_line(self) -> None:
        buf = io.StringIO()
        buf.isatty = lambda: True  # type: ignore[assignment]

        with patch("shard.ui.status.sys.stderr", buf):
            with StatusFeed() as status:
                status.update("Something...")
            # __exit__ calls clear()

        output = buf.getvalue()
        # Should end with a clear-line sequence
        assert output.endswith("\r\033[K")


class TestStatusFeedNonTTY:
    """Verify non-TTY makes update() a no-op."""

    def test_non_tty_produces_no_output(self) -> None:
        buf = io.StringIO()
        buf.isatty = lambda: False  # type: ignore[assignment]

        with patch("shard.ui.status.sys.stderr", buf):
            with StatusFeed() as status:
                status.update("Should not appear")
                status.update("Nor this")

        assert buf.getvalue() == ""

    def test_no_isatty_method_is_no_op(self) -> None:
        """stderr without isatty (e.g. some CI environments)."""

        class NoTTYStream:
            """A stream-like object with no isatty method."""

            def __init__(self) -> None:
                self.data = ""

            def write(self, s: str) -> int:
                self.data += s
                return len(s)

            def flush(self) -> None:
                pass

        buf = NoTTYStream()

        with patch("shard.ui.status.sys.stderr", buf):
            with StatusFeed() as status:
                status.update("Should not appear")

        assert buf.data == ""


class TestStatusFeedTruncation:
    """Verify long messages are truncated to terminal width."""

    def test_long_message_truncated(self) -> None:
        buf = io.StringIO()
        buf.isatty = lambda: True  # type: ignore[assignment]

        with (
            patch("shard.ui.status.sys.stderr", buf),
            patch("shard.ui.status.shutil.get_terminal_size", return_value=os.terminal_size((40, 24))),
        ):
            with StatusFeed() as status:
                status.update("A" * 100)

        output = buf.getvalue()
        # Find the content after the ANSI clear sequence
        content_start = output.find("\r\033[K") + len("\r\033[K")
        content_end = output.find("\r\033[K", content_start)
        if content_end == -1:
            content = output[content_start:]
        else:
            content = output[content_start:content_end]
        # Must not exceed terminal width - 2
        assert len(content) <= 38


class TestStatusFeedSpinnerCycle:
    """Verify spinner cycles through characters correctly."""

    def test_spinner_cycles(self) -> None:
        buf = io.StringIO()
        buf.isatty = lambda: True  # type: ignore[assignment]

        with patch("shard.ui.status.sys.stderr", buf):
            with StatusFeed() as status:
                for _ in range(len(_SPINNER_CHARS) + 2):
                    status.update("test")

        output = buf.getvalue()
        # Each spinner char should appear at least once
        for char in _SPINNER_CHARS:
            assert char in output

    def test_first_spinner_is_first_char(self) -> None:
        buf = io.StringIO()
        buf.isatty = lambda: True  # type: ignore[assignment]

        with patch("shard.ui.status.sys.stderr", buf):
            with StatusFeed() as status:
                status.update("test")

        output = buf.getvalue()
        # After \r\033[K, first char should be the first spinner
        content_start = output.find("\r\033[K") + len("\r\033[K")
        assert output[content_start] == _SPINNER_CHARS[0]
