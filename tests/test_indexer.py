"""Tests for shard.pipeline.indexer — chunk_text word-boundary correctness,
overlap semantics, edge cases, and index_note (with mocked ChromaDB).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shard.config import ShardConfig
from shard.pipeline import FormattedNote, IndexingError, SourceType
from shard.pipeline.indexer import chunk_text, index_note

# ── chunk_text ────────────────────────────────────────────────────────────────


class TestChunkText:
    # ── basic splitting ───────────────────────────────────────────────────

    def test_empty_string_returns_empty_list(self) -> None:
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert chunk_text("   \n\t  ") == []

    def test_text_shorter_than_chunk_size_returns_single_chunk(self) -> None:
        text = "word " * 10  # 10 words
        chunks = chunk_text(text, chunk_size=500)

        assert len(chunks) == 1
        # All original words should be present
        assert set(chunks[0].split()) == set(text.split())

    def test_single_chunk_contains_all_words(self) -> None:
        text = "alpha beta gamma delta epsilon"
        chunks = chunk_text(text, chunk_size=100)

        assert chunks == ["alpha beta gamma delta epsilon"]

    def test_exact_chunk_size_produces_single_chunk(self) -> None:
        words = ["word"] * 5
        chunks = chunk_text(" ".join(words), chunk_size=5, overlap=0)

        assert len(chunks) == 1

    # ── chunking with no overlap ──────────────────────────────────────────

    def test_two_chunks_with_zero_overlap(self) -> None:
        # 6 words, chunk_size=3, overlap=0 → 2 non-overlapping chunks
        text = "a b c d e f"
        chunks = chunk_text(text, chunk_size=3, overlap=0)

        assert len(chunks) == 2
        assert chunks[0] == "a b c"
        assert chunks[1] == "d e f"

    def test_chunks_cover_all_words_with_zero_overlap(self) -> None:
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=5, overlap=0)

        recovered = []
        for chunk in chunks:
            recovered.extend(chunk.split())

        assert recovered == words

    # ── chunking with overlap ─────────────────────────────────────────────

    def test_overlap_words_shared_between_consecutive_chunks(self) -> None:
        # 10 words, chunk_size=6, overlap=2, step=4
        # starts: 0, 4, 8  →  3 chunks
        # chunk 0: words 0-5, chunk 1: words 4-9 — last 2 of chunk 0 == first 2 of chunk 1
        words = [f"w{i}" for i in range(10)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=6, overlap=2)

        assert len(chunks) == 3
        # Last 2 words of chunk 0 should equal first 2 words of chunk 1
        tail_0 = chunks[0].split()[-2:]
        head_1 = chunks[1].split()[:2]
        assert tail_0 == head_1

    def test_overlap_produces_correct_chunk_count(self) -> None:
        # 10 words, chunk_size=5, overlap=2, step=3
        # starts: 0, 3, 6, 9  →  4 chunks (last chunk is a single word partial)
        words = [f"w{i}" for i in range(10)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=5, overlap=2)

        assert len(chunks) == 4

    def test_all_input_words_appear_in_at_least_one_chunk(self) -> None:
        words = [f"w{i}" for i in range(15)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=5, overlap=2)

        all_chunk_words: set[str] = set()
        for chunk in chunks:
            all_chunk_words.update(chunk.split())

        assert all_chunk_words == set(words)

    # ── word-boundary preservation ────────────────────────────────────────

    def test_chunks_do_not_split_mid_word(self) -> None:
        text = "The quick brown fox jumps over the lazy dog and then some more words here"
        chunks = chunk_text(text, chunk_size=5, overlap=1)

        for chunk in chunks:
            # Every element must be a non-empty "word" (no partial words)
            for word in chunk.split():
                assert word  # no empty strings
                assert " " not in word  # words themselves have no spaces

    def test_chunk_boundaries_align_with_whitespace(self) -> None:
        words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=3, overlap=1)

        for chunk in chunks:
            for word in chunk.split():
                assert word in words

    # ── overlap clamping ──────────────────────────────────────────────────

    def test_overlap_clamped_when_equal_to_chunk_size(self) -> None:
        # overlap == chunk_size would cause infinite loop without clamping
        text = "a b c d e f g h i j"
        # Should not hang; guard clamps overlap to chunk_size - 1
        chunks = chunk_text(text, chunk_size=3, overlap=3)

        assert len(chunks) > 0

    def test_overlap_clamped_when_greater_than_chunk_size(self) -> None:
        text = "a b c d e f g h i j"
        chunks = chunk_text(text, chunk_size=3, overlap=10)

        assert len(chunks) > 0

    # ── default parameters ────────────────────────────────────────────────

    def test_default_parameters_produce_chunks(self) -> None:
        # Default: chunk_size=500, overlap=50
        words = ["word"] * 600
        text = " ".join(words)
        chunks = chunk_text(text)

        # With 600 words, chunk_size=500, overlap=50, step=450
        # starts: 0, 450 → 2 chunks
        assert len(chunks) == 2

    def test_default_chunk_size_500_words_each(self) -> None:
        words = ["word"] * 600
        text = " ".join(words)
        chunks = chunk_text(text)

        assert len(chunks[0].split()) == 500


# ── index_note (mocked ChromaDB) ──────────────────────────────────────────────


def _make_formatted_note(
    title: str = "Test Note",
    body: str = "alpha beta gamma delta epsilon",
    tags: list[str] | None = None,
) -> FormattedNote:
    return FormattedNote(
        title=title,
        tags=tags or ["test", "indexer"],
        summary="A test note summary.",
        body=body,
        source="https://example.com",
        source_type=SourceType.URL,
    )


class TestIndexNote:
    def _make_mock_collection(self) -> MagicMock:
        collection = MagicMock()
        collection.upsert = MagicMock()
        return collection

    def test_returns_indexed_note_with_correct_title(
        self, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        note = _make_formatted_note(title="My Note")
        path = tmp_path / "my-note.md"
        mock_collection = self._make_mock_collection()

        with patch("shard.pipeline.indexer._get_collection", return_value=mock_collection):
            result = index_note(note, path, mock_config)

        assert result.title == "My Note"

    def test_returns_indexed_note_with_correct_tags(
        self, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        note = _make_formatted_note(tags=["python", "testing"])
        path = tmp_path / "note.md"
        mock_collection = self._make_mock_collection()

        with patch("shard.pipeline.indexer._get_collection", return_value=mock_collection):
            result = index_note(note, path, mock_config)

        assert result.tags == ["python", "testing"]

    def test_num_chunks_matches_chunks_produced(
        self, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        # 10 words, chunk_size=500 (default) → 1 chunk
        note = _make_formatted_note(body="one two three four five six seven eight nine ten")
        path = tmp_path / "note.md"
        mock_collection = self._make_mock_collection()

        with patch("shard.pipeline.indexer._get_collection", return_value=mock_collection):
            result = index_note(note, path, mock_config)

        assert result.num_chunks == 1

    def test_upsert_called_with_correct_document_count(
        self, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        # Build a body with enough words to guarantee 2 chunks at defaults
        words = ["word"] * 600
        note = _make_formatted_note(body=" ".join(words))
        path = tmp_path / "big-note.md"
        mock_collection = self._make_mock_collection()

        with patch("shard.pipeline.indexer._get_collection", return_value=mock_collection):
            result = index_note(note, path, mock_config)

        # Verify upsert was called once and documents count matches num_chunks
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args
        documents = call_kwargs.kwargs.get("documents") or call_kwargs.args[1]
        assert len(documents) == result.num_chunks

    def test_chunk_ids_use_path_stem(
        self, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        note = _make_formatted_note()
        path = tmp_path / "my-note-stem.md"
        mock_collection = self._make_mock_collection()

        with patch("shard.pipeline.indexer._get_collection", return_value=mock_collection):
            index_note(note, path, mock_config)

        call_kwargs = mock_collection.upsert.call_args
        ids = call_kwargs.kwargs.get("ids") or call_kwargs.args[0]
        for doc_id in ids:
            assert doc_id.startswith("my-note-stem_chunk_")

    def test_empty_body_uses_summary_as_fallback_chunk(
        self, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        note = FormattedNote(
            title="Empty Body Note",
            tags=[],
            summary="A meaningful summary.",
            body="",
            source="text",
            source_type=SourceType.TEXT,
        )
        path = tmp_path / "empty-body.md"
        mock_collection = self._make_mock_collection()

        with patch("shard.pipeline.indexer._get_collection", return_value=mock_collection):
            result = index_note(note, path, mock_config)

        assert result.num_chunks == 1
        call_kwargs = mock_collection.upsert.call_args
        documents = call_kwargs.kwargs.get("documents") or call_kwargs.args[1]
        assert documents[0] == "A meaningful summary."

    def test_empty_body_and_summary_falls_back_to_title(
        self, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        note = FormattedNote(
            title="Title Only Note",
            tags=[],
            summary="",
            body="",
            source="text",
            source_type=SourceType.TEXT,
        )
        path = tmp_path / "title-only.md"
        mock_collection = self._make_mock_collection()

        with patch("shard.pipeline.indexer._get_collection", return_value=mock_collection):
            result = index_note(note, path, mock_config)

        assert result.num_chunks == 1
        call_kwargs = mock_collection.upsert.call_args
        documents = call_kwargs.kwargs.get("documents") or call_kwargs.args[1]
        assert documents[0] == "Title Only Note"

    def test_metadata_passed_to_upsert_has_required_keys(
        self, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        note = _make_formatted_note()
        path = tmp_path / "meta-check.md"
        mock_collection = self._make_mock_collection()

        with patch("shard.pipeline.indexer._get_collection", return_value=mock_collection):
            index_note(note, path, mock_config)

        call_kwargs = mock_collection.upsert.call_args
        metadatas = call_kwargs.kwargs.get("metadatas") or call_kwargs.args[2]
        meta = metadatas[0]

        assert "title" in meta
        assert "source" in meta
        assert "source_type" in meta
        assert "tags" in meta
        assert "path" in meta

    def test_chromadb_error_wrapped_as_indexing_error(
        self, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        note = _make_formatted_note()
        path = tmp_path / "error-note.md"
        mock_collection = self._make_mock_collection()
        mock_collection.upsert.side_effect = RuntimeError("ChromaDB exploded")

        with patch("shard.pipeline.indexer._get_collection", return_value=mock_collection):
            with pytest.raises(IndexingError, match="Failed to index"):
                index_note(note, path, mock_config)
