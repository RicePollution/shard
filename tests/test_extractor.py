"""Tests for shard.pipeline.extractor — auto-detection, PDF detection, YouTube ID parsing,
plain-text extraction, and empty-input error handling.

All tests that would require network access or real file parsing are covered by
mocking the respective dependencies.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shard.pipeline import ExtractionError, SourceType
from shard.pipeline.extractor import (
    _extract_text,
    _looks_like_pdf,
    _parse_youtube_video_id,
    extract,
)

# ── _looks_like_pdf ───────────────────────────────────────────────────────────


class TestLooksLikePdf:
    def test_path_with_pdf_extension_returns_true(self) -> None:
        assert _looks_like_pdf("/home/user/document.pdf") is True

    def test_path_with_pdf_extension_case_insensitive(self) -> None:
        assert _looks_like_pdf("/home/user/document.PDF") is True
        assert _looks_like_pdf("/home/user/document.Pdf") is True

    def test_http_url_returns_false_even_with_pdf_extension(self) -> None:
        assert _looks_like_pdf("https://example.com/file.pdf") is False

    def test_http_url_returns_false(self) -> None:
        assert _looks_like_pdf("http://example.com/page") is False

    def test_plain_text_path_returns_false(self) -> None:
        assert _looks_like_pdf("some plain text") is False

    def test_txt_extension_returns_false(self) -> None:
        assert _looks_like_pdf("/home/user/notes.txt") is False

    def test_non_existent_pdf_extension_returns_true(self, tmp_path: Path) -> None:
        # Extension check happens before existence check
        non_existent = tmp_path / "ghost.pdf"
        assert _looks_like_pdf(str(non_existent)) is True

    def test_file_with_pdf_magic_bytes_returns_true(self, tmp_path: Path) -> None:
        fake_pdf = tmp_path / "noPdfExtension"
        fake_pdf.write_bytes(b"%PDF-1.4 fake content")
        assert _looks_like_pdf(str(fake_pdf)) is True

    def test_file_without_pdf_magic_bytes_returns_false(self, tmp_path: Path) -> None:
        not_pdf = tmp_path / "notaPdf"
        not_pdf.write_bytes(b"PLAINTEXT content here")
        assert _looks_like_pdf(str(not_pdf)) is False

    def test_oversized_string_returns_false(self) -> None:
        # Strings longer than 4096 chars should not be treated as file paths
        long_string = "a.pdf" + "x" * 4100
        assert _looks_like_pdf(long_string) is False


# ── _parse_youtube_video_id ───────────────────────────────────────────────────


class TestParseYoutubeVideoId:
    def test_standard_watch_url(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert _parse_youtube_video_id(url) == "dQw4w9WgXcQ"

    def test_short_youtu_be_url(self) -> None:
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert _parse_youtube_video_id(url) == "dQw4w9WgXcQ"

    def test_embed_url(self) -> None:
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert _parse_youtube_video_id(url) == "dQw4w9WgXcQ"

    def test_shorts_url(self) -> None:
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert _parse_youtube_video_id(url) == "dQw4w9WgXcQ"

    def test_watch_url_with_extra_query_params(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PL123"
        assert _parse_youtube_video_id(url) == "dQw4w9WgXcQ"

    def test_non_youtube_url_returns_none(self) -> None:
        assert _parse_youtube_video_id("https://vimeo.com/123456789") is None

    def test_completely_invalid_string_returns_none(self) -> None:
        assert _parse_youtube_video_id("not a url at all") is None

    def test_www_youtu_be_prefix(self) -> None:
        url = "https://www.youtu.be/dQw4w9WgXcQ"
        assert _parse_youtube_video_id(url) == "dQw4w9WgXcQ"

    def test_video_id_with_hyphens_and_underscores(self) -> None:
        url = "https://www.youtube.com/watch?v=A-B_C1234de"
        assert _parse_youtube_video_id(url) == "A-B_C1234de"


# ── _extract_text ─────────────────────────────────────────────────────────────


class TestExtractText:
    def test_returns_extracted_content_with_correct_text(self) -> None:
        result = _extract_text("Hello, this is some text.")

        assert result.text == "Hello, this is some text."
        assert result.source == "text"
        assert result.source_type == SourceType.TEXT

    def test_title_is_first_line_of_input(self) -> None:
        result = _extract_text("First Line\nSecond Line\nThird Line")

        assert result.title == "First Line"

    def test_title_is_truncated_to_120_chars(self) -> None:
        long_line = "W" * 200
        result = _extract_text(long_line)

        assert len(result.title) == 120

    def test_metadata_includes_char_count(self) -> None:
        text = "Short text."
        result = _extract_text(text)

        assert result.metadata["char_count"] == str(len(text))

    def test_raises_extraction_error_on_empty_string(self) -> None:
        with pytest.raises(ExtractionError, match="empty"):
            _extract_text("")

    def test_raises_extraction_error_on_whitespace_only(self) -> None:
        with pytest.raises(ExtractionError, match="empty"):
            _extract_text("   \n\t  ")


# ── extract() auto-detection ──────────────────────────────────────────────────


class TestExtractAutoDetection:
    def test_dispatches_to_text_extractor_for_plain_input(self) -> None:
        result = extract("Some plain text content here.")

        assert result.source_type == SourceType.TEXT
        assert result.text == "Some plain text content here."

    def test_dispatches_to_pdf_extractor_for_pdf_path(self, tmp_path: Path) -> None:
        fake_pdf = tmp_path / "report.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF page content"
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extract(str(fake_pdf))

        assert result.source_type == SourceType.PDF

    def test_dispatches_to_youtube_for_youtube_com_url(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        mock_snippet = MagicMock()
        mock_snippet.text = "Hello from transcript"
        mock_transcript = MagicMock()
        mock_transcript.__iter__ = MagicMock(return_value=iter([mock_snippet]))
        mock_transcript.language_code = "en"
        mock_transcript.is_generated = False
        mock_api_instance = MagicMock()
        mock_api_instance.fetch.return_value = mock_transcript
        mock_api_class = MagicMock(return_value=mock_api_instance)

        with (
            patch("shard.pipeline.extractor._fetch_youtube_title", return_value="Test Video"),
            # YouTubeTranscriptApi is imported inside _extract_youtube; patch at
            # the source module where it is defined.
            patch("youtube_transcript_api.YouTubeTranscriptApi", mock_api_class, create=True),
            patch(
                "shard.pipeline.extractor._extract_youtube",
                return_value=MagicMock(
                    source_type=SourceType.YOUTUBE,
                    text="Hello from transcript",
                    source=url,
                    title="Test Video",
                    metadata={},
                ),
            ),
        ):
            result = extract(url)

        assert result.source_type == SourceType.YOUTUBE

    def test_dispatches_to_youtube_for_youtu_be_url(self) -> None:
        url = "https://youtu.be/dQw4w9WgXcQ"

        with patch(
            "shard.pipeline.extractor._extract_youtube",
            return_value=MagicMock(
                source_type=SourceType.YOUTUBE,
                text="Transcript snippet",
                source=url,
                title="dQw4w9WgXcQ",
                metadata={},
            ),
        ):
            result = extract(url)

        assert result.source_type == SourceType.YOUTUBE

    def test_dispatches_to_url_extractor_for_http_url(self) -> None:
        url = "https://example.com/article"

        mock_response = MagicMock()
        mock_response.text = "<html><body><article>Article body text</article></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            result = extract(url)

        assert result.source_type == SourceType.URL

    def test_strips_whitespace_before_detection(self) -> None:
        result = extract("  Hello plain text  ")

        assert result.source_type == SourceType.TEXT

    def test_raises_extraction_error_for_empty_text(self) -> None:
        # When stdin is a tty (normal terminal), empty input falls through to
        # _extract_text which raises ExtractionError.  Force isatty() to True
        # so pytest's captured stdin does not interfere.
        with patch("shard.pipeline.extractor.sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with pytest.raises(ExtractionError):
                extract("")

    def test_pdf_not_found_raises_extraction_error(self, tmp_path: Path) -> None:
        missing_pdf = tmp_path / "missing.pdf"
        # File does not exist but has .pdf extension — _looks_like_pdf returns True

        with pytest.raises(ExtractionError, match="not found"):
            extract(str(missing_pdf))
