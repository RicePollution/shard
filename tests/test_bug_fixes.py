"""Regression tests for bug fixes (GitHub issues #1–#5).

Each test class corresponds to one issue and verifies the fix works correctly.
"""

from __future__ import annotations

import json
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shard.config import ShardConfig
from shard.pipeline import FormattedNote, FormattingError, IndexingError, SourceType


# ── Issue #1: litellm.completion timeout ─────────────────────────────────────


class TestCompletionTimeout:
    """Verify that complete() passes a timeout to litellm.completion."""

    def test_complete_passes_timeout_kwarg(self) -> None:
        from shard.models import _COMPLETION_TIMEOUT, complete

        fake_config = MagicMock()
        fake_config.model = "ollama_chat/qwen2.5:3b"

        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with (
            patch("shard.models.get_config", return_value=fake_config),
            patch("shard.models.inject_api_keys"),
            patch("litellm.completion", return_value=mock_response) as mock_completion,
        ):
            complete("test prompt", model="ollama_chat/qwen2.5:3b")

        call_kwargs = mock_completion.call_args.kwargs
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] == _COMPLETION_TIMEOUT

    def test_completion_timeout_is_positive_integer(self) -> None:
        from shard.models import _COMPLETION_TIMEOUT

        assert isinstance(_COMPLETION_TIMEOUT, int)
        assert _COMPLETION_TIMEOUT > 0


# ── Issue #2: YouTube transcript timeout ─────────────────────────────────────


class TestYouTubeTranscriptTimeout:
    """Verify that _extract_youtube injects a session with timeout."""

    def test_youtube_api_receives_session_with_timeout(self) -> None:
        """The YouTubeTranscriptApi should be instantiated with an http_client."""
        from unittest.mock import ANY

        mock_snippet = MagicMock()
        mock_snippet.text = "Hello transcript"
        mock_transcript = MagicMock()
        mock_transcript.__iter__ = MagicMock(return_value=iter([mock_snippet]))
        mock_transcript.language_code = "en"
        mock_transcript.is_generated = False

        mock_api_instance = MagicMock()
        mock_api_instance.fetch.return_value = mock_transcript
        mock_api_class = MagicMock(return_value=mock_api_instance)

        mock_errors = MagicMock()
        mock_errors.YouTubeTranscriptApiException = Exception

        with (
            patch("shard.pipeline.extractor._fetch_youtube_title", return_value="Test"),
            patch.dict("sys.modules", {
                "youtube_transcript_api": MagicMock(YouTubeTranscriptApi=mock_api_class),
                "youtube_transcript_api._errors": mock_errors,
            }),
            patch("shard.pipeline.extractor.requests", create=True),
        ):
            from shard.pipeline.extractor import _extract_youtube

            _extract_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        # Verify YouTubeTranscriptApi was called with an http_client argument
        mock_api_class.assert_called_once()
        call_kwargs = mock_api_class.call_args
        assert call_kwargs.kwargs.get("http_client") is not None or (
            len(call_kwargs.args) > 0 and call_kwargs.args[0] is not None
        )


# ── Issue #3: Embedding model download timeout ──────────────────────────────


class TestEmbeddingModelTimeout:
    """Verify that _load_embedding_fn times out instead of hanging."""

    def test_load_embedding_fn_raises_on_timeout(self) -> None:
        from shard.pipeline.indexer import _load_embedding_fn

        def _hang(*args: object, **kwargs: object) -> None:
            import time
            time.sleep(10)  # simulate a stall

        with patch(
            "shard.pipeline.indexer.SentenceTransformerEmbeddingFunction",
            side_effect=_hang,
        ):
            with patch("shard.pipeline.indexer._EMBEDDING_LOAD_TIMEOUT", 0.1):
                with pytest.raises(IndexingError, match="Timed out"):
                    _load_embedding_fn("all-MiniLM-L6-v2")

    def test_load_embedding_fn_returns_on_success(self) -> None:
        from shard.pipeline.indexer import _load_embedding_fn

        mock_fn = MagicMock()
        with patch(
            "shard.pipeline.indexer.SentenceTransformerEmbeddingFunction",
            return_value=mock_fn,
        ):
            result = _load_embedding_fn("all-MiniLM-L6-v2")

        assert result is mock_fn


# ── Issue #4: Collection reuse in runner loop ────────────────────────────────


class TestCollectionReuse:
    """Verify that index_note accepts and uses a pre-built collection."""

    def test_index_note_uses_passed_collection(
        self, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        from shard.pipeline.indexer import index_note

        note = FormattedNote(
            title="Test", tags=["test"], summary="Summary",
            body="some words here", source="text", source_type=SourceType.TEXT,
        )
        path = tmp_path / "test.md"
        mock_collection = MagicMock()

        # _get_collection should NOT be called when a collection is passed
        with patch("shard.pipeline.indexer._get_collection") as mock_get:
            index_note(note, path, mock_config, collection=mock_collection)

        mock_get.assert_not_called()
        mock_collection.upsert.assert_called_once()

    def test_index_note_falls_back_to_get_collection(
        self, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        from shard.pipeline.indexer import index_note

        note = FormattedNote(
            title="Test", tags=["test"], summary="Summary",
            body="some words here", source="text", source_type=SourceType.TEXT,
        )
        path = tmp_path / "test.md"
        mock_collection = MagicMock()

        with patch("shard.pipeline.indexer._get_collection", return_value=mock_collection):
            index_note(note, path, mock_config)

        mock_collection.upsert.assert_called_once()

    def test_runner_builds_collection_once(self) -> None:
        """run_add_pipeline should call _get_collection once, not per note."""
        from shard.pipeline import ExtractedContent
        from shard.runner import run_add_pipeline

        mock_config = MagicMock()
        mock_config.vault_path = Path("/tmp/vault")
        mock_config.notes_subfolder = ""

        mock_formatted = FormattedNote(
            title="Note", tags=["t"], summary="S",
            body="body text", source="text", source_type=SourceType.TEXT,
        )

        mock_extracted = MagicMock(spec=ExtractedContent)
        mock_extracted.source_type = SourceType.TEXT
        mock_extracted.title = "Test"

        mock_indexed = MagicMock()
        mock_indexed.num_chunks = 1

        mock_collection = MagicMock()

        with (
            patch("shard.runner.get_config", return_value=mock_config),
            patch("shard.runner.extract", return_value=mock_extracted),
            patch("shard.runner.format_notes", return_value=[mock_formatted, mock_formatted]),
            patch("shard.runner.save_note", return_value=Path("/tmp/vault/note.md")),
            patch("shard.runner._get_collection", return_value=mock_collection) as mock_get_coll,
            patch("shard.runner.index_note", return_value=mock_indexed) as mock_index,
        ):
            run_add_pipeline("test input", config=mock_config)

        # _get_collection called exactly once, not per note
        mock_get_coll.assert_called_once()
        # index_note called twice (once per note), each with the collection
        assert mock_index.call_count == 2
        for call in mock_index.call_args_list:
            assert call.kwargs.get("collection") is mock_collection


# ── Issue #5: JSON fence stripping and parent_summary guard ──────────────────


class TestJsonFenceStripping:
    """Verify _parse_json_response handles trailing newlines after fences."""

    def test_fence_with_trailing_newline(self) -> None:
        from shard.pipeline.formatter import _parse_json_response

        raw = '```json\n{"key": "value"}\n```\n'
        result = _parse_json_response(raw)
        assert result == {"key": "value"}

    def test_fence_with_multiple_trailing_newlines(self) -> None:
        from shard.pipeline.formatter import _parse_json_response

        raw = '```json\n{"key": "value"}\n```\n\n\n'
        result = _parse_json_response(raw)
        assert result == {"key": "value"}

    def test_fence_without_trailing_newline_still_works(self) -> None:
        from shard.pipeline.formatter import _parse_json_response

        raw = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(raw)
        assert result == {"key": "value"}

    def test_fence_with_trailing_spaces(self) -> None:
        from shard.pipeline.formatter import _parse_json_response

        raw = '```json\n{"key": "value"}\n```   '
        result = _parse_json_response(raw)
        assert result == {"key": "value"}


class TestParentSummaryValidation:
    """Verify Stage A raises FormattingError when parent_summary is missing."""

    def test_missing_parent_summary_raises_error(self) -> None:
        from shard.pipeline import ExtractedContent
        from shard.pipeline.formatter import _stage_a_decompose

        extracted = ExtractedContent(
            text="Some text", source="test", source_type=SourceType.URL,
            title="Test", metadata={},
        )
        # Has parent_topic and subtopics but no parent_summary
        response = json.dumps({
            "parent_topic": "Topic",
            "subtopics": [{"title": "Sub", "focus": "f", "relevant_section": "r"}],
        })
        with patch("shard.pipeline.formatter.complete", return_value=response):
            with pytest.raises(FormattingError, match="parent_summary"):
                _stage_a_decompose("some text", extracted)

    def test_present_parent_summary_passes_validation(self) -> None:
        from shard.pipeline import ExtractedContent
        from shard.pipeline.formatter import _stage_a_decompose

        extracted = ExtractedContent(
            text="Some text", source="test", source_type=SourceType.URL,
            title="Test", metadata={},
        )
        response = json.dumps({
            "parent_topic": "Topic",
            "parent_summary": "A summary.",
            "subtopics": [{"title": "Sub", "focus": "f", "relevant_section": "r"}],
        })
        with patch("shard.pipeline.formatter.complete", return_value=response):
            result = _stage_a_decompose("some text", extracted)

        assert result["parent_summary"] == "A summary."


# ── Ollama pull timeout ──────────────────────────────────────────────────────


class TestOllamaPullTimeout:
    """Verify that pull_ollama_model uses a bounded timeout, not None."""

    def test_pull_uses_bounded_timeout(self) -> None:
        from shard.models import pull_ollama_model

        with patch("shard.models.httpx.stream") as mock_stream:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.iter_lines = MagicMock(return_value=iter([]))
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_stream.return_value = mock_response

            pull_ollama_model("test-model")

        call_kwargs = mock_stream.call_args.kwargs
        timeout = call_kwargs.get("timeout")
        # Must not be None (was the bug)
        assert timeout is not None
