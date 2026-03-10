"""Tests for shard.pipeline.linker — Linker, apply_links, and helper functions."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from shard.pipeline import LinkError
from shard.pipeline.linker import (
    LinkSuggestion,
    Linker,
    _parse_link_response,
    _replace_first_outside_links,
    apply_links,
)

# ── Fixtures and shared helpers ───────────────────────────────────────────────

_VALID_JSON_RESPONSE = json.dumps([
    {
        "original_text": "machine learning",
        "linked_text": "[[Machine Learning|machine learning]]",
        "note_title": "Machine Learning",
    }
])

_NOTE_CONTENT = """\
---
title: 'Test Note'
tags:
  - 'testing'
date: '2024-01-15'
---

This note discusses machine learning and Python programming.
"""


def _make_suggestion(
    original: str = "machine learning",
    linked: str = "[[Machine Learning|machine learning]]",
    title: str = "Machine Learning",
) -> LinkSuggestion:
    return LinkSuggestion(
        original_text=original,
        linked_text=linked,
        note_title=title,
    )


# ── Linker.find_links ─────────────────────────────────────────────────────────


class TestFindLinks:
    def test_find_links_returns_suggestions(self) -> None:
        linker = Linker()
        with patch("shard.pipeline.linker.complete", return_value=_VALID_JSON_RESPONSE):
            results = linker.find_links(_NOTE_CONTENT, ["Machine Learning", "Python"])

        assert len(results) == 1
        assert isinstance(results[0], LinkSuggestion)
        assert results[0].original_text == "machine learning"
        assert results[0].linked_text == "[[Machine Learning|machine learning]]"
        assert results[0].note_title == "Machine Learning"

    def test_find_links_empty_titles_returns_empty_list_without_model_call(
        self,
    ) -> None:
        linker = Linker()
        mock_complete = MagicMock()

        with patch("shard.pipeline.linker.complete", mock_complete):
            results = linker.find_links(_NOTE_CONTENT, [])

        assert results == []
        mock_complete.assert_not_called()

    def test_find_links_raises_link_error_on_model_failure(self) -> None:
        linker = Linker()
        with patch(
            "shard.pipeline.linker.complete",
            side_effect=Exception("Connection refused"),
        ):
            with pytest.raises(LinkError, match="Model call failed"):
                linker.find_links(_NOTE_CONTENT, ["Machine Learning"])

    def test_find_links_returns_multiple_suggestions(self) -> None:
        response = json.dumps([
            {
                "original_text": "machine learning",
                "linked_text": "[[Machine Learning|machine learning]]",
                "note_title": "Machine Learning",
            },
            {
                "original_text": "Python",
                "linked_text": "[[Python|Python]]",
                "note_title": "Python",
            },
        ])
        linker = Linker()
        with patch("shard.pipeline.linker.complete", return_value=response):
            results = linker.find_links(_NOTE_CONTENT, ["Machine Learning", "Python"])

        assert len(results) == 2
        assert results[1].original_text == "Python"

    def test_find_links_returns_empty_list_on_empty_json_array(self) -> None:
        linker = Linker()
        with patch("shard.pipeline.linker.complete", return_value="[]"):
            results = linker.find_links(_NOTE_CONTENT, ["Machine Learning"])

        assert results == []


# ── apply_links ───────────────────────────────────────────────────────────────


class TestApplyLinks:
    def test_existing_wikilinks_not_double_linked(self) -> None:
        content = "See [[Machine Learning|machine learning]] for details."
        suggestion = _make_suggestion(
            original="machine learning",
            linked="[[Machine Learning|machine learning]]",
        )
        result = apply_links(content, [suggestion])

        # The existing wikilink text must remain unchanged and not be nested.
        assert result == content
        assert "[[[[" not in result

    def test_apply_links_first_occurrence_only(self) -> None:
        content = "Python is great. I love Python for data work."
        suggestion = _make_suggestion(
            original="Python",
            linked="[[Python|Python]]",
            title="Python",
        )
        result = apply_links(content, [suggestion])

        assert result.count("[[Python|Python]]") == 1
        # Second occurrence remains as plain text.
        assert "I love Python for data work" in result

    def test_apply_links_preserves_markdown_links(self) -> None:
        content = "Read the [Python docs](https://docs.python.org) for more info."
        suggestion = _make_suggestion(
            original="Python docs",
            linked="[[Python|Python docs]]",
            title="Python",
        )
        result = apply_links(content, [suggestion])

        # The markdown link must not be rewritten.
        assert "[Python docs](https://docs.python.org)" in result

    def test_apply_links_substitutes_plain_occurrence(self) -> None:
        content = "This note discusses machine learning concepts."
        suggestion = _make_suggestion()
        result = apply_links(content, [suggestion])

        assert "[[Machine Learning|machine learning]]" in result
        # Original plain text is gone.
        assert "discusses machine learning concepts" not in result

    def test_apply_links_no_match_leaves_content_unchanged(self) -> None:
        content = "This note has nothing relevant."
        suggestion = _make_suggestion(original="deep learning")
        result = apply_links(content, [suggestion])

        assert result == content

    def test_apply_links_applies_multiple_suggestions(self) -> None:
        content = "machine learning and Python are both important topics."
        suggestions = [
            _make_suggestion(
                original="machine learning",
                linked="[[Machine Learning|machine learning]]",
            ),
            _make_suggestion(
                original="Python",
                linked="[[Python|Python]]",
                title="Python",
            ),
        ]
        result = apply_links(content, suggestions)

        assert "[[Machine Learning|machine learning]]" in result
        assert "[[Python|Python]]" in result

    def test_words_under_4_chars_not_linked_via_apply_links(self) -> None:
        # apply_links is not responsible for length filtering — the LLM prompt
        # handles that — but it must faithfully apply whatever suggestions it
        # receives.  A short-word suggestion is still applied if provided.
        content = "The cat sat on the mat."
        suggestion = _make_suggestion(original="cat", linked="[[Cat|cat]]", title="Cat")
        result = apply_links(content, [suggestion])

        # The replacement is applied because apply_links does not filter by length.
        assert "[[Cat|cat]]" in result


# ── _replace_first_outside_links ──────────────────────────────────────────────


class TestReplaceFirstOutsideLinks:
    def test_replaces_plain_text_occurrence(self) -> None:
        content = "Python is popular."
        result = _replace_first_outside_links(content, "Python", "[[Python|Python]]")

        assert result == "[[Python|Python]] is popular."

    def test_skips_occurrence_inside_wikilink(self) -> None:
        content = "See [[Python|Python]] for more."
        result = _replace_first_outside_links(content, "Python", "[[Python|Python]]")

        # No change — the only occurrence is inside existing brackets.
        assert result == content

    def test_skips_occurrence_inside_markdown_link(self) -> None:
        content = "Visit [Python](https://python.org) today."
        result = _replace_first_outside_links(content, "Python", "[[Python|Python]]")

        assert result == content

    def test_replaces_first_unprotected_when_mixed(self) -> None:
        # First occurrence is inside [[]], second is plain text.
        content = "Loved [[Python|Python]] and also Python for scripting."
        result = _replace_first_outside_links(content, "Python", "[[Python|Python]]")

        # The second (plain) occurrence should be replaced.
        assert result.count("[[Python|Python]]") == 2
        assert "also [[Python|Python]] for scripting" in result

    def test_returns_content_unchanged_when_original_absent(self) -> None:
        content = "No relevant content here."
        result = _replace_first_outside_links(content, "Python", "[[Python|Python]]")

        assert result == content

    def test_replaces_only_first_of_multiple_plain_occurrences(self) -> None:
        content = "Python and Python and Python."
        result = _replace_first_outside_links(content, "Python", "[[Python|Python]]")

        assert result == "[[Python|Python]] and Python and Python."


# ── _parse_link_response ──────────────────────────────────────────────────────


class TestParseLinkResponse:
    def test_parses_plain_json_array(self) -> None:
        results = _parse_link_response(_VALID_JSON_RESPONSE)

        assert len(results) == 1
        assert results[0].original_text == "machine learning"

    def test_strips_json_code_fences(self) -> None:
        fenced = f"```json\n{_VALID_JSON_RESPONSE}\n```"
        results = _parse_link_response(fenced)

        assert len(results) == 1
        assert results[0].note_title == "Machine Learning"

    def test_strips_plain_code_fences(self) -> None:
        fenced = f"```\n{_VALID_JSON_RESPONSE}\n```"
        results = _parse_link_response(fenced)

        assert len(results) == 1

    def test_raises_link_error_on_garbage_input(self) -> None:
        with pytest.raises(LinkError, match="Failed to parse"):
            _parse_link_response("this is not json at all %%$#")

    def test_raises_link_error_on_empty_string(self) -> None:
        with pytest.raises(LinkError):
            _parse_link_response("")

    def test_skips_items_with_missing_fields(self) -> None:
        # Item is missing "note_title" — should be silently skipped.
        response = json.dumps([
            {
                "original_text": "machine learning",
                "linked_text": "[[Machine Learning|machine learning]]",
                # note_title deliberately omitted
            }
        ])
        results = _parse_link_response(response)

        assert results == []

    def test_skips_non_dict_items_in_array(self) -> None:
        response = json.dumps(["not a dict", 42, None])
        results = _parse_link_response(response)

        assert results == []

    def test_extracts_json_embedded_in_prose(self) -> None:
        prose = (
            "Sure, here are the suggestions:\n"
            + _VALID_JSON_RESPONSE
            + "\nHope that helps!"
        )
        results = _parse_link_response(prose)

        assert len(results) == 1
        assert results[0].original_text == "machine learning"

    def test_returns_empty_list_for_empty_array(self) -> None:
        results = _parse_link_response("[]")

        assert results == []
