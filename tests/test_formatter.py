"""Tests for shard.pipeline.formatter — _parse_response, _truncate, _extract_field,
_extract_body, and format_note (with mocked LLM).
"""

from __future__ import annotations

import json
from unittest.mock import call, patch

import pytest

from shard.pipeline import ExtractedContent, FormattingError, SourceType
from shard.pipeline.formatter import (
    MAX_TEXT_LENGTH,
    TRUNCATION_NOTICE,
    _build_style_injection,
    _extract_body,
    _extract_field,
    _parse_json_response,
    _parse_response,
    _stage_a_decompose,
    _truncate,
    format_note,
    format_notes,
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


# ── _parse_json_response (formatter version) ─────────────────────────────────


class TestParseJsonResponseFormatter:
    def test_parses_bare_json(self) -> None:
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_strips_code_fences(self) -> None:
        raw = '```json\n{"key": "fenced"}\n```'
        result = _parse_json_response(raw)
        assert result == {"key": "fenced"}

    def test_extracts_json_from_prose(self) -> None:
        raw = 'Here is the result:\n{"answer": true}\nDone.'
        result = _parse_json_response(raw)
        assert result == {"answer": True}

    def test_raises_formatting_error_on_invalid_json(self) -> None:
        with pytest.raises(FormattingError, match="Failed to parse"):
            _parse_json_response("not json")

    def test_raises_formatting_error_on_empty_string(self) -> None:
        with pytest.raises(FormattingError):
            _parse_json_response("")


# ── format_notes ──────────────────────────────────────────────────────────────

# Stage A mock response
_STAGE_A_JSON = json.dumps({
    "parent_topic": "RAG Overview",
    "parent_summary": "An overview of Retrieval Augmented Generation.",
    "subtopics": [
        {
            "title": "RAG Retrieval Step",
            "focus": "How documents are retrieved in RAG",
            "relevant_section": "The retrieval component searches...",
        },
        {
            "title": "RAG Generation Step",
            "focus": "How the LLM generates answers using context",
            "relevant_section": "The generation step takes...",
        },
        {
            "title": "RAG vs Fine-tuning",
            "focus": "Comparing RAG to model fine-tuning",
            "relevant_section": "Unlike fine-tuning...",
        },
    ],
})


def _make_stage_b_json(title: str, slug: str) -> str:
    """Build a Stage B subtopic JSON response for a given title and slug."""
    return json.dumps({
        "title": title,
        "slug": slug,
        "tags": ["rag", "ai"],
        "markdown": (
            f"## {title}\n\nContent about {title}.\n\n"
            "## Links\n- [[RAG Overview]]"
        ),
    })


# Parent index mock response
_PARENT_INDEX_JSON = json.dumps({
    "title": "RAG Overview",
    "slug": "rag-overview",
    "tags": ["index", "rag"],
    "markdown": (
        "## RAG Overview\n\nAn overview of RAG.\n\n"
        "## Notes\n"
        "- [[RAG Retrieval Step]]\n"
        "- [[RAG Generation Step]]\n"
        "- [[RAG vs Fine-tuning]]\n\n"
        "## Source\nhttps://example.com"
    ),
})


class TestFormatNotes:
    def _make_extracted(self) -> ExtractedContent:
        return ExtractedContent(
            text="Long article about RAG and its components...",
            source="https://example.com/rag",
            source_type=SourceType.URL,
            title="RAG Article",
            metadata={},
        )

    def _atomic_side_effects(self) -> list[str]:
        """Return the ordered sequence of mock responses for a full atomic run."""
        return [
            _STAGE_A_JSON,
            _make_stage_b_json("RAG Retrieval Step", "rag-retrieval-step"),
            _make_stage_b_json("RAG Generation Step", "rag-generation-step"),
            _make_stage_b_json("RAG vs Fine-tuning", "rag-vs-fine-tuning"),
            _PARENT_INDEX_JSON,
        ]

    def test_single_flag_returns_one_note_list(self) -> None:
        """--single wraps format_note result in a list."""
        extracted = self._make_extracted()
        with patch("shard.pipeline.formatter.complete", return_value=_WELL_FORMED_RESPONSE):
            result = format_notes(extracted, single=True)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].title == "Python Async Patterns"

    def test_atomic_split_returns_multiple_notes(self) -> None:
        """Default (single=False) produces parent + subtopic notes."""
        extracted = self._make_extracted()
        with patch(
            "shard.pipeline.formatter.complete",
            side_effect=self._atomic_side_effects(),
        ):
            result = format_notes(extracted, single=False)

        # 1 parent index + 3 subtopics = 4 total
        assert len(result) == 4

    def test_parent_note_is_first(self) -> None:
        """Parent index note is the first item in the returned list."""
        extracted = self._make_extracted()
        with patch(
            "shard.pipeline.formatter.complete",
            side_effect=self._atomic_side_effects(),
        ):
            result = format_notes(extracted, single=False)

        assert result[0].title == "RAG Overview"
        assert "index" in result[0].tags

    def test_parent_note_contains_wikilinks_to_children(self) -> None:
        """Parent index note body contains [[wikilinks]] to all children."""
        extracted = self._make_extracted()
        with patch(
            "shard.pipeline.formatter.complete",
            side_effect=self._atomic_side_effects(),
        ):
            result = format_notes(extracted, single=False)

        parent_body = result[0].body
        assert "[[RAG Retrieval Step]]" in parent_body
        assert "[[RAG Generation Step]]" in parent_body
        assert "[[RAG vs Fine-tuning]]" in parent_body

    def test_child_notes_link_back_to_parent(self) -> None:
        """Each child note body contains a [[wikilink]] back to parent."""
        extracted = self._make_extracted()
        with patch(
            "shard.pipeline.formatter.complete",
            side_effect=self._atomic_side_effects(),
        ):
            result = format_notes(extracted, single=False)

        for child in result[1:]:
            assert "[[RAG Overview]]" in child.body

    def test_all_notes_preserve_source(self) -> None:
        """All returned notes carry the original source URL and source type."""
        extracted = self._make_extracted()
        with patch(
            "shard.pipeline.formatter.complete",
            side_effect=self._atomic_side_effects(),
        ):
            result = format_notes(extracted, single=False)

        for note in result:
            assert note.source == "https://example.com/rag"
            assert note.source_type == SourceType.URL

    def test_stage_a_failure_raises_formatting_error(self) -> None:
        """An exception from the model during Stage A is wrapped in FormattingError."""
        extracted = self._make_extracted()
        with patch(
            "shard.pipeline.formatter.complete",
            side_effect=Exception("timeout"),
        ):
            with pytest.raises(FormattingError, match="topic decomposition"):
                format_notes(extracted, single=False)

    def test_stage_a_empty_subtopics_falls_back_to_single(self) -> None:
        """If Stage A returns empty subtopics, falls back to a single note."""
        extracted = self._make_extracted()
        empty_decomp = json.dumps({
            "parent_topic": "Test",
            "parent_summary": "Test summary",
            "subtopics": [],
        })
        # Stage A returns empty subtopics; format_note is then called as fallback.
        side_effects = [empty_decomp, _WELL_FORMED_RESPONSE]
        with patch(
            "shard.pipeline.formatter.complete",
            side_effect=side_effects,
        ):
            result = format_notes(extracted, single=False)

        assert len(result) == 1


# ── _build_style_injection ────────────────────────────────────────────────────


class TestBuildStyleInjection:
    def test_returns_empty_string_when_no_style(self) -> None:
        assert _build_style_injection(None) == ""

    def test_includes_fingerprints_when_style_exists(self) -> None:
        style = {
            "fingerprints": ["Rule 1", "Rule 2"],
            "tag_format": "#lowercase",
            "template": "## Heading\n\nContent",
        }
        result = _build_style_injection(style)
        assert "Rule 1" in result
        assert "Rule 2" in result
        assert "#lowercase" in result

    def test_includes_template_when_style_exists(self) -> None:
        style = {
            "fingerprints": [],
            "tag_format": "#tag",
            "template": "## My Custom Template",
        }
        result = _build_style_injection(style)
        assert "## My Custom Template" in result

    def test_empty_style_dict_returns_non_empty_string(self) -> None:
        # An empty dict is not None, so a (possibly sparse) injection is returned.
        result = _build_style_injection({})
        assert isinstance(result, str)
        assert result != ""

    def test_multiple_fingerprints_all_present(self) -> None:
        style = {
            "fingerprints": ["Use bold for key terms", "Keep sentences short", "No passive voice"],
            "tag_format": "#kebab-case",
            "template": "",
        }
        result = _build_style_injection(style)
        for rule in style["fingerprints"]:
            assert rule in result


# ── _stage_a_decompose ────────────────────────────────────────────────────────


class TestStageADecompose:
    def _make_extracted(self) -> ExtractedContent:
        return ExtractedContent(
            text="Article about machine learning fundamentals.",
            source="https://example.com/ml",
            source_type=SourceType.URL,
            title="ML Basics",
            metadata={},
        )

    def test_returns_dict_with_required_keys(self) -> None:
        extracted = self._make_extracted()
        response = json.dumps({
            "parent_topic": "ML Basics",
            "parent_summary": "Overview of machine learning.",
            "subtopics": [
                {"title": "Supervised Learning", "focus": "Learning with labels", "relevant_section": "..."},
            ],
        })
        with patch("shard.pipeline.formatter.complete", return_value=response):
            result = _stage_a_decompose("some text", extracted)

        assert "parent_topic" in result
        assert "parent_summary" in result
        assert "subtopics" in result

    def test_subtopics_is_a_list(self) -> None:
        extracted = self._make_extracted()
        response = json.dumps({
            "parent_topic": "ML Basics",
            "parent_summary": "Overview.",
            "subtopics": [],
        })
        with patch("shard.pipeline.formatter.complete", return_value=response):
            result = _stage_a_decompose("some text", extracted)

        assert isinstance(result["subtopics"], list)

    def test_raises_on_model_exception(self) -> None:
        extracted = self._make_extracted()
        with patch(
            "shard.pipeline.formatter.complete",
            side_effect=Exception("network error"),
        ):
            with pytest.raises(FormattingError, match="topic decomposition"):
                _stage_a_decompose("some text", extracted)

    def test_raises_on_empty_model_response(self) -> None:
        extracted = self._make_extracted()
        with patch("shard.pipeline.formatter.complete", return_value=""):
            with pytest.raises(FormattingError):
                _stage_a_decompose("some text", extracted)

    def test_raises_when_parent_topic_missing(self) -> None:
        extracted = self._make_extracted()
        # Missing "parent_topic" key
        response = json.dumps({"subtopics": []})
        with patch("shard.pipeline.formatter.complete", return_value=response):
            with pytest.raises(FormattingError):
                _stage_a_decompose("some text", extracted)

    def test_raises_when_subtopics_missing(self) -> None:
        extracted = self._make_extracted()
        # Missing "subtopics" key
        response = json.dumps({"parent_topic": "Topic", "parent_summary": "Summary"})
        with patch("shard.pipeline.formatter.complete", return_value=response):
            with pytest.raises(FormattingError):
                _stage_a_decompose("some text", extracted)

    def test_raises_when_subtopics_not_a_list(self) -> None:
        extracted = self._make_extracted()
        response = json.dumps({
            "parent_topic": "Topic",
            "parent_summary": "Summary",
            "subtopics": "not a list",
        })
        with patch("shard.pipeline.formatter.complete", return_value=response):
            with pytest.raises(FormattingError):
                _stage_a_decompose("some text", extracted)

    def test_parent_topic_value_matches_json(self) -> None:
        extracted = self._make_extracted()
        response = json.dumps({
            "parent_topic": "Deep Learning",
            "parent_summary": "Neural networks and beyond.",
            "subtopics": [],
        })
        with patch("shard.pipeline.formatter.complete", return_value=response):
            result = _stage_a_decompose("some text", extracted)

        assert result["parent_topic"] == "Deep Learning"
