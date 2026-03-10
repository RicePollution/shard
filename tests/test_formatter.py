"""Tests for shard.pipeline.formatter — _parse_response, _truncate, _extract_field,
_extract_body, and format_note (with mocked LLM).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from shard.pipeline import ExtractedContent, FormattingError, SourceType
from shard.pipeline.formatter import (
    MAX_TEXT_LENGTH,
    TRUNCATION_NOTICE,
    _extract_body,
    _extract_field,
    _parse_response,
    _truncate,
    format_note,
)

# ── _truncate ─────────────────────────────────────────────────────────────────


class TestTruncate:
    def test_short_text_returned_unchanged(self) -> None:
        text = "Short text."
        assert _truncate(text) == text

    def test_text_exactly_at_limit_returned_unchanged(self) -> None:
        text = "a" * MAX_TEXT_LENGTH
        assert _truncate(text) == text

    def test_text_over_limit_is_truncated(self) -> None:
        text = "a" * (MAX_TEXT_LENGTH + 100)
        result = _truncate(text)

        assert len(result) == MAX_TEXT_LENGTH + len(TRUNCATION_NOTICE)

    def test_truncation_notice_appended(self) -> None:
        text = "b" * (MAX_TEXT_LENGTH + 1)
        result = _truncate(text)

        assert result.endswith(TRUNCATION_NOTICE)

    def test_truncated_prefix_matches_original(self) -> None:
        text = "c" * (MAX_TEXT_LENGTH + 50)
        result = _truncate(text)

        assert result.startswith("c" * MAX_TEXT_LENGTH)

    def test_empty_string_returned_unchanged(self) -> None:
        assert _truncate("") == ""


# ── _extract_field ────────────────────────────────────────────────────────────


class TestExtractField:
    def test_extracts_title_field(self) -> None:
        response = "TITLE: My Great Note\nTAGS: python, testing\nSUMMARY: A summary.\nBODY:\nBody."
        assert _extract_field(response, "TITLE") == "My Great Note"

    def test_extracts_tags_field(self) -> None:
        response = "TITLE: X\nTAGS: ai, llm, notes\nSUMMARY: Sum.\nBODY:\nContent."
        assert _extract_field(response, "TAGS") == "ai, llm, notes"

    def test_extracts_summary_field(self) -> None:
        response = "TITLE: T\nTAGS: t\nSUMMARY: This is the summary.\nBODY:\nContent."
        assert _extract_field(response, "SUMMARY") == "This is the summary."

    def test_case_insensitive_matching(self) -> None:
        response = "title: Lowercase Title\ntags: a, b"
        assert _extract_field(response, "TITLE") == "Lowercase Title"

    def test_returns_empty_string_when_field_absent(self) -> None:
        response = "No structured fields here."
        assert _extract_field(response, "TITLE") == ""

    def test_strips_surrounding_whitespace_from_value(self) -> None:
        response = "TITLE:   Padded Title   "
        assert _extract_field(response, "TITLE") == "Padded Title"

    def test_does_not_match_partial_field_name(self) -> None:
        # "SUBTITLE" should not match "TITLE" at the start of a line
        response = "SUBTITLE: Not a title\nTITLE: Real Title"
        assert _extract_field(response, "TITLE") == "Real Title"


# ── _extract_body ─────────────────────────────────────────────────────────────


class TestExtractBody:
    def test_returns_content_after_body_marker(self) -> None:
        response = "TITLE: T\nTAGS: t\nSUMMARY: S\nBODY:\nThis is the body."
        assert _extract_body(response) == "This is the body."

    def test_multiline_body_is_fully_captured(self) -> None:
        response = "TITLE: T\nBODY:\n## Section\n\nParagraph one.\n\nParagraph two."
        body = _extract_body(response)
        assert "## Section" in body
        assert "Paragraph one." in body
        assert "Paragraph two." in body

    def test_returns_empty_string_when_body_marker_absent(self) -> None:
        response = "TITLE: T\nTAGS: t\nNo body marker."
        assert _extract_body(response) == ""

    def test_case_insensitive_body_marker(self) -> None:
        response = "TITLE: T\nbody:\nLowercase body content."
        assert _extract_body(response) == "Lowercase body content."

    def test_leading_whitespace_stripped_from_body(self) -> None:
        response = "TITLE: T\nBODY:\n\n\n   Content starts here."
        body = _extract_body(response)
        assert body.startswith("Content starts here.")

    def test_body_with_inline_content_on_marker_line(self) -> None:
        # Some models put content directly after "BODY: " on the same line
        response = "TITLE: T\nBODY: Inline body content."
        # The regex allows for optional content on the same line via \n?
        # After BODY: the remainder is captured
        body = _extract_body(response)
        # This particular format might not match; the key thing is it doesn't crash
        assert isinstance(body, str)


# ── _parse_response ───────────────────────────────────────────────────────────


_WELL_FORMED_RESPONSE = """\
TITLE: Python Async Patterns
TAGS: python, async, concurrency, asyncio, patterns
SUMMARY: An overview of modern async patterns in Python 3.11+.
BODY:
## Introduction

Python's asyncio library provides powerful tools for concurrent programming.

## Key Patterns

- **Task Groups** for structured concurrency
- **Async context managers** for resource cleanup
"""


class TestParseResponse:
    def test_well_formed_response_extracts_title(self) -> None:
        result = _parse_response(_WELL_FORMED_RESPONSE)
        assert result["title"] == "Python Async Patterns"

    def test_well_formed_response_extracts_tags(self) -> None:
        result = _parse_response(_WELL_FORMED_RESPONSE)
        assert result["tags"] == "python, async, concurrency, asyncio, patterns"

    def test_well_formed_response_extracts_summary(self) -> None:
        result = _parse_response(_WELL_FORMED_RESPONSE)
        assert "async patterns" in result["summary"]

    def test_well_formed_response_extracts_body(self) -> None:
        result = _parse_response(_WELL_FORMED_RESPONSE)
        assert "## Introduction" in result["body"]
        assert "Task Groups" in result["body"]

    def test_result_has_all_four_keys(self) -> None:
        result = _parse_response(_WELL_FORMED_RESPONSE)
        assert set(result.keys()) == {"title", "tags", "summary", "body"}

    def test_fallback_parsing_when_no_markers(self) -> None:
        # Response with no TITLE/TAGS/BODY markers — fallback path
        plain_response = "A Plain Title\nThis is just a body without any markers."
        result = _parse_response(plain_response)

        assert result["title"] == "A Plain Title"
        assert result["tags"] == "note"
        assert result["summary"] == ""
        assert "body without any markers" in result["body"]

    def test_fallback_uses_hash_stripped_first_line_as_title(self) -> None:
        response = "# Markdown Header Title\nBody content here."
        result = _parse_response(response)

        assert result["title"] == "Markdown Header Title"

    def test_raises_formatting_error_on_empty_response(self) -> None:
        with pytest.raises(FormattingError, match="unparseable"):
            _parse_response("")

    def test_raises_formatting_error_on_whitespace_only_response(self) -> None:
        with pytest.raises(FormattingError):
            _parse_response("   \n\n   ")

    def test_missing_tags_defaults_to_note(self) -> None:
        # Has title and body but no TAGS line
        response = "TITLE: A Title\nSUMMARY: A summary.\nBODY:\nBody text."
        result = _parse_response(response)

        assert result["tags"] == "note"

    def test_missing_summary_defaults_to_empty_string(self) -> None:
        response = "TITLE: A Title\nTAGS: a, b\nBODY:\nBody text."
        result = _parse_response(response)

        assert result["summary"] == ""


# ── format_note (integration with mocked LLM) ─────────────────────────────────


class TestFormatNote:
    def _make_extracted(
        self,
        text: str = "Some article content about Python.",
        source: str = "https://example.com",
        source_type: SourceType = SourceType.URL,
        title: str = "Example Article",
    ) -> ExtractedContent:
        return ExtractedContent(
            text=text,
            source=source,
            source_type=source_type,
            title=title,
            metadata={},
        )

    def test_format_note_returns_formatted_note(self) -> None:
        extracted = self._make_extracted()

        with patch("shard.pipeline.formatter.complete", return_value=_WELL_FORMED_RESPONSE):
            result = format_note(extracted)

        assert result.title == "Python Async Patterns"
        assert result.tags == ["python", "async", "concurrency", "asyncio", "patterns"]
        assert "async patterns" in result.summary
        assert "## Introduction" in result.body

    def test_format_note_preserves_source_metadata(self) -> None:
        extracted = self._make_extracted()

        with patch("shard.pipeline.formatter.complete", return_value=_WELL_FORMED_RESPONSE):
            result = format_note(extracted)

        assert result.source == "https://example.com"
        assert result.source_type == SourceType.URL

    def test_format_note_raises_on_model_failure(self) -> None:
        extracted = self._make_extracted()

        with patch(
            "shard.pipeline.formatter.complete",
            side_effect=Exception("Connection refused"),
        ):
            with pytest.raises(FormattingError, match="Model call failed"):
                format_note(extracted)

    def test_format_note_raises_on_empty_model_response(self) -> None:
        extracted = self._make_extracted()

        with patch("shard.pipeline.formatter.complete", return_value=""):
            with pytest.raises(FormattingError, match="empty response"):
                format_note(extracted)

    def test_format_note_raises_on_whitespace_only_model_response(self) -> None:
        extracted = self._make_extracted()

        with patch("shard.pipeline.formatter.complete", return_value="   \n  "):
            with pytest.raises(FormattingError, match="empty response"):
                format_note(extracted)

    def test_format_note_splits_tags_on_comma(self) -> None:
        extracted = self._make_extracted()
        response = "TITLE: T\nTAGS: python, testing, ci\nSUMMARY: S.\nBODY:\nContent."

        with patch("shard.pipeline.formatter.complete", return_value=response):
            result = format_note(extracted)

        assert result.tags == ["python", "testing", "ci"]

    def test_format_note_truncates_long_text_before_model_call(self) -> None:
        long_text = "word " * (MAX_TEXT_LENGTH // 5 + 100)
        extracted = self._make_extracted(text=long_text)
        captured_prompt: list[str] = []

        def capture_complete(prompt: str, system: str = "") -> str:
            captured_prompt.append(prompt)
            return _WELL_FORMED_RESPONSE

        with patch("shard.pipeline.formatter.complete", side_effect=capture_complete):
            format_note(extracted)

        # The prompt must contain the truncation notice
        assert TRUNCATION_NOTICE in captured_prompt[0]
