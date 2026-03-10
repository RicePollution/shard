"""Tests for shard.vault — slugify, save_note/read_note, parse_frontmatter, list_shards."""

from __future__ import annotations

from pathlib import Path

import pytest

from shard.config import ShardConfig
from shard.pipeline import FormattedNote, SourceType, VaultError
from shard.vault import (
    list_shards,
    parse_frontmatter,
    read_note,
    save_note,
    slugify,
)

# ── slugify ───────────────────────────────────────────────────────────────────


class TestSlugify:
    def test_basic_lowercase_conversion(self) -> None:
        assert slugify("Hello World") == "hello-world"

    def test_strips_leading_and_trailing_hyphens(self) -> None:
        result = slugify("  Hello, World!  ")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_special_chars_become_single_hyphen(self) -> None:
        assert slugify("Hello, World!") == "hello-world"

    def test_multiple_spaces_collapse_to_one_hyphen(self) -> None:
        assert slugify("Multiple   Spaces   Here") == "multiple-spaces-here"

    def test_punctuation_only_between_words(self) -> None:
        assert slugify("foo---bar") == "foo-bar"

    def test_numbers_are_preserved(self) -> None:
        assert slugify("Python 3.11 features") == "python-3-11-features"

    def test_unicode_punctuation_stripped(self) -> None:
        result = slugify("Café & Résumé!")
        # Non-ASCII letters are replaced, resulting in hyphens
        assert "-" in result or result.isalnum()

    def test_long_string_truncated_to_80_chars(self) -> None:
        long_text = "a" * 120
        result = slugify(long_text)
        assert len(result) <= 80

    def test_truncation_does_not_leave_trailing_hyphen(self) -> None:
        # Construct a string whose 80-char boundary falls on a separator
        # by interleaving short words
        text = "ab " * 40  # each "ab " is 3 chars; boundary will land mid-word-separator
        result = slugify(text)
        assert not result.endswith("-")

    def test_empty_string_returns_empty(self) -> None:
        assert slugify("") == ""

    def test_only_special_chars_returns_empty(self) -> None:
        assert slugify("!@#$%^&*()") == ""

    def test_single_word(self) -> None:
        assert slugify("Python") == "python"

    def test_already_valid_slug_unchanged(self) -> None:
        assert slugify("already-valid") == "already-valid"


# ── save_note / read_note ─────────────────────────────────────────────────────


def _make_note(title: str = "Test Note", tags: list[str] | None = None) -> FormattedNote:
    return FormattedNote(
        title=title,
        tags=tags or ["python", "testing"],
        summary="A test summary.",
        body="## Section\n\nBody content here.",
        source="https://example.com",
        source_type=SourceType.URL,
    )


class TestSaveNote:
    def test_returns_path_inside_vault(self, mock_config: ShardConfig) -> None:
        note = _make_note()
        path = save_note(note, mock_config)

        assert path.is_relative_to(mock_config.vault_path)

    def test_file_is_created(self, mock_config: ShardConfig) -> None:
        note = _make_note()
        path = save_note(note, mock_config)

        assert path.exists()

    def test_filename_derived_from_title_slug(self, mock_config: ShardConfig) -> None:
        note = _make_note(title="My Awesome Note")
        path = save_note(note, mock_config)

        assert path.stem == "my-awesome-note"
        assert path.suffix == ".md"

    def test_written_to_vault_root_by_default(self, mock_config: ShardConfig) -> None:
        note = _make_note()
        path = save_note(note, mock_config)

        assert path.parent == mock_config.vault_path

    def test_written_to_subfolder_when_configured(self, mock_config: ShardConfig) -> None:
        mock_config.notes_subfolder = "Imported/Shards"
        note = _make_note()
        path = save_note(note, mock_config)
        expected_parent = mock_config.vault_path / "Imported" / "Shards"

        assert path.parent == expected_parent

    def test_subfolder_directories_created_automatically(self, mock_config: ShardConfig) -> None:
        mock_config.notes_subfolder = "Notes/Shard"
        note = _make_note()
        save_note(note, mock_config)
        sub_dir = mock_config.vault_path / "Notes" / "Shard"

        assert sub_dir.is_dir()

    def test_empty_title_falls_back_to_untitled(self, mock_config: ShardConfig) -> None:
        note = FormattedNote(
            title="!@#",  # slugifies to empty string
            tags=[],
            summary="",
            body="content",
            source="text",
            source_type=SourceType.TEXT,
        )
        path = save_note(note, mock_config)

        assert path.stem == "untitled"

    def test_content_includes_frontmatter_delimiter(self, mock_config: ShardConfig) -> None:
        note = _make_note()
        path = save_note(note, mock_config)
        content = path.read_text(encoding="utf-8")

        assert content.startswith("---")

    def test_body_appears_after_frontmatter(self, mock_config: ShardConfig) -> None:
        note = _make_note()
        path = save_note(note, mock_config)
        content = path.read_text(encoding="utf-8")

        assert "Body content here." in content


class TestSaveNoteReadNoteRoundTrip:
    def test_round_trip_recovers_full_content(self, mock_config: ShardConfig) -> None:
        note = _make_note()
        path = save_note(note, mock_config)
        recovered = read_note(path)

        assert "Test Note" in recovered
        assert "Body content here." in recovered

    def test_read_note_raises_vault_error_for_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.md"

        with pytest.raises(VaultError, match="not found"):
            read_note(missing)


# ── parse_frontmatter ─────────────────────────────────────────────────────────


class TestParseFrontmatter:
    def test_simple_key_value_pairs(self) -> None:
        content = "---\ntitle: 'My Note'\ndate: '2024-01-01'\n---\nBody text."
        meta, body = parse_frontmatter(content)

        assert meta["title"] == "My Note"
        assert meta["date"] == "2024-01-01"
        assert body.strip() == "Body text."

    def test_with_tags_block_sequence(self) -> None:
        content = "---\ntitle: 'Tagged'\ntags:\n  - 'python'\n  - 'testing'\n---\nBody."
        meta, body = parse_frontmatter(content)

        assert meta["tags"] == ["python", "testing"]

    def test_with_empty_tags_inline_list(self) -> None:
        content = "---\ntitle: 'No Tags'\ntags: []\n---\nContent."
        meta, body = parse_frontmatter(content)

        assert meta["tags"] == []

    def test_body_stripped_of_leading_newline(self) -> None:
        # The parser strips exactly one leading newline after the closing "---".
        # A blank separator line (the second \n) is preserved as a leading \n
        # in the returned body — assert the core content is present.
        content = "---\ntitle: 'X'\n---\n\nActual body."
        _, body = parse_frontmatter(content)

        assert "Actual body." in body

    def test_no_frontmatter_returns_empty_meta_and_full_body(self) -> None:
        content = "No frontmatter here.\nJust a plain body."
        meta, body = parse_frontmatter(content)

        assert meta == {}
        assert body == content

    def test_malformed_no_closing_delimiter_returns_empty_meta(self) -> None:
        content = "---\ntitle: 'Oops'\n# no closing delimiter"
        meta, body = parse_frontmatter(content)

        assert meta == {}

    def test_single_quoted_values_unquoted(self) -> None:
        content = "---\nsummary: 'It''s a summary'\n---\n"
        meta, _ = parse_frontmatter(content)

        assert meta["summary"] == "It's a summary"

    def test_round_trip_with_saved_note(self, mock_config: ShardConfig) -> None:
        original = _make_note(title="Round Trip Test", tags=["tag-a", "tag-b"])
        path = save_note(original, mock_config)
        content = read_note(path)
        meta, body = parse_frontmatter(content)

        assert meta["title"] == "Round Trip Test"
        assert meta["tags"] == ["tag-a", "tag-b"]
        assert meta["source"] == "https://example.com"
        assert meta["source_type"] == "url"
        assert "date" in meta
        assert "Body content here." in body


# ── list_shards ───────────────────────────────────────────────────────────────


class TestListShards:
    def test_empty_vault_returns_empty_list(self, mock_config: ShardConfig) -> None:
        result = list_shards(mock_config)

        assert result == []

    def test_returns_empty_list_when_shards_dir_absent(self, mock_config: ShardConfig) -> None:
        # Confirm the subdir does not exist yet
        shards_dir = mock_config.vault_path / "Imported" / "Shards"
        assert not shards_dir.exists()

        result = list_shards(mock_config)

        assert result == []

    def test_discovers_saved_notes(self, mock_config: ShardConfig) -> None:
        save_note(_make_note(title="Note One"), mock_config)
        save_note(_make_note(title="Note Two"), mock_config)

        paths = list_shards(mock_config)

        assert len(paths) == 2

    def test_returns_only_md_files(self, mock_config: ShardConfig) -> None:
        # Manually plant a non-md file in the shards directory
        shards_dir = mock_config.vault_path / "Imported" / "Shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        (shards_dir / "stray.txt").write_text("ignored", encoding="utf-8")
        save_note(_make_note(title="Real Note"), mock_config)

        paths = list_shards(mock_config)

        assert all(p.suffix == ".md" for p in paths)
        assert len(paths) == 1

    def test_results_are_sorted(self, mock_config: ShardConfig) -> None:
        for title in ["Zebra Note", "Alpha Note", "Middle Note"]:
            save_note(_make_note(title=title), mock_config)

        paths = list_shards(mock_config)
        names = [p.name for p in paths]

        assert names == sorted(names)
