"""Content extraction from various sources."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from shard.pipeline import ExtractedContent, ExtractionError, SourceType

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract(input_str: str) -> ExtractedContent:
    """Auto-detect the input type and dispatch to the appropriate extractor.

    Detection order:
    1. PDF  — path ends in ``.pdf`` or the file exists and has a PDF magic
              bytes header.
    2. YouTube — URL contains ``youtube.com`` or ``youtu.be``.
    3. URL  — starts with ``http://`` or ``https://``.
    4. stdin — ``sys.stdin`` is not a tty AND ``input_str`` is empty or ``"-"``.
    5. Text — fallback: treat ``input_str`` as raw text.

    Args:
        input_str: A file path, URL, ``"-"``, empty string, or raw text.

    Returns:
        An :class:`~shard.pipeline.ExtractedContent` instance.

    Raises:
        ExtractionError: If extraction fails for any detected source type.
    """
    stripped = input_str.strip()

    # --- PDF -----------------------------------------------------------------
    if _looks_like_pdf(stripped):
        return _extract_pdf(stripped)

    # --- YouTube -------------------------------------------------------------
    if "youtube.com" in stripped or "youtu.be" in stripped:
        return _extract_youtube(stripped)

    # --- Generic URL ---------------------------------------------------------
    if stripped.startswith(("http://", "https://")):
        return _extract_url(stripped)

    # --- stdin ----------------------------------------------------------------
    if not sys.stdin.isatty() and (stripped == "" or stripped == "-"):
        return _extract_stdin()

    # --- Plain text fallback -------------------------------------------------
    return _extract_text(stripped)


# ---------------------------------------------------------------------------
# Source-specific extractors
# ---------------------------------------------------------------------------


def _extract_pdf(path: str) -> ExtractedContent:
    """Extract text from a PDF file using pdfplumber.

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        :class:`~shard.pipeline.ExtractedContent` with text from all pages.

    Raises:
        ExtractionError: If the file cannot be opened or parsed.
    """
    try:
        import pdfplumber  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ExtractionError("pdfplumber is not installed.") from exc

    pdf_path = Path(path)

    if not pdf_path.exists():
        raise ExtractionError(f"PDF file not found: {path}")

    try:
        pages: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text: str | None = page.extract_text()
                if page_text:
                    pages.append(page_text)
    except Exception as exc:
        raise ExtractionError(f"Failed to read PDF '{path}': {exc}") from exc

    if not pages:
        raise ExtractionError(f"No extractable text found in PDF: {path}")

    full_text = "\n\n".join(pages)

    # Derive title: first non-empty line of the document, or the filename stem.
    first_line = next((ln.strip() for ln in full_text.splitlines() if ln.strip()), "")
    title = first_line[:120] if first_line else pdf_path.stem

    return ExtractedContent(
        text=full_text,
        source=str(pdf_path.resolve()),
        source_type=SourceType.PDF,
        title=title,
        metadata={"pages": str(len(pages)), "filename": pdf_path.name},
    )


def _extract_url(url: str) -> ExtractedContent:
    """Fetch a web page and extract its readable text content.

    Uses ``httpx`` for the HTTP request and ``beautifulsoup4`` for parsing.
    Prefers ``<article>`` over ``<main>`` over ``<body>`` for the content
    container.

    Args:
        url: A fully-qualified HTTP/HTTPS URL.

    Returns:
        :class:`~shard.pipeline.ExtractedContent` with the page's text.

    Raises:
        ExtractionError: On network error, HTTP error, or parse failure.
    """
    try:
        import httpx  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ExtractionError("httpx is not installed.") from exc

    try:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ExtractionError("beautifulsoup4 is not installed.") from exc

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(
                url,
                headers={"User-Agent": "shard-cli/0.1 (+https://github.com/shard-cli)"},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ExtractionError(
            f"HTTP {exc.response.status_code} fetching URL: {url}"
        ) from exc
    except httpx.RequestError as exc:
        raise ExtractionError(f"Network error fetching URL '{url}': {exc}") from exc

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as exc:
        raise ExtractionError(f"Failed to parse HTML from '{url}': {exc}") from exc

    # Extract title
    title_tag = soup.find("title")
    title: str = title_tag.get_text(strip=True) if title_tag else url

    # Remove script / style / nav noise before extracting text
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Prefer the most semantically specific content container
    content_tag = soup.find("article") or soup.find("main") or soup.find("body")
    if content_tag is None:
        raise ExtractionError(f"Could not find any content container in '{url}'.")

    # Collapse whitespace from text chunks while preserving paragraph breaks
    paragraphs = [
        chunk.strip()
        for chunk in content_tag.get_text(separator="\n").splitlines()
        if chunk.strip()
    ]
    text = "\n".join(paragraphs)

    if not text:
        raise ExtractionError(f"No text content extracted from '{url}'.")

    return ExtractedContent(
        text=text,
        source=url,
        source_type=SourceType.URL,
        title=title,
        metadata={"content_length": str(len(text))},
    )


def _extract_youtube(url: str) -> ExtractedContent:
    """Fetch and join a YouTube video transcript.

    Supports both ``youtube.com/watch?v=<ID>`` and ``youtu.be/<ID>`` URL
    formats.  Uses the v2 :class:`YouTubeTranscriptApi` by instantiating the
    class and calling ``.fetch(video_id)``.

    Args:
        url: A YouTube video URL.

    Returns:
        :class:`~shard.pipeline.ExtractedContent` with the transcript text.

    Raises:
        ExtractionError: If the video ID cannot be parsed or the transcript
            cannot be fetched.
    """
    try:
        from youtube_transcript_api import (  # type: ignore[import-untyped]
            YouTubeTranscriptApi,
        )
        from youtube_transcript_api._errors import (  # type: ignore[import-untyped]
            YouTubeTranscriptApiException,
        )
    except ImportError as exc:
        raise ExtractionError("youtube-transcript-api is not installed.") from exc

    video_id = _parse_youtube_video_id(url)
    if not video_id:
        raise ExtractionError(f"Could not parse a YouTube video ID from URL: {url}")

    # Attempt to obtain the page title via a lightweight HTTP fetch so that
    # notes have a human-readable title rather than a bare video ID.
    title = _fetch_youtube_title(url) or video_id

    try:
        import requests

        session = requests.Session()
        original_request = session.request

        def _request_with_timeout(method: str, url: str, **kw: object) -> object:
            kw.setdefault("timeout", 30)
            return original_request(method, url, **kw)

        session.request = _request_with_timeout  # type: ignore[assignment]
        api = YouTubeTranscriptApi(http_client=session)
        transcript = api.fetch(video_id)
    except YouTubeTranscriptApiException as exc:
        raise ExtractionError(
            f"Could not fetch transcript for YouTube video '{video_id}': {exc}"
        ) from exc
    except Exception as exc:
        raise ExtractionError(
            f"Unexpected error fetching YouTube transcript for '{video_id}': {exc}"
        ) from exc

    snippets = [snippet.text.strip() for snippet in transcript if snippet.text.strip()]
    if not snippets:
        raise ExtractionError(f"Transcript for YouTube video '{video_id}' is empty.")

    text = " ".join(snippets)

    return ExtractedContent(
        text=text,
        source=url,
        source_type=SourceType.YOUTUBE,
        title=title,
        metadata={
            "video_id": video_id,
            "language": transcript.language_code,
            "is_generated": str(transcript.is_generated),
            "snippet_count": str(len(snippets)),
        },
    )


def _extract_stdin() -> ExtractedContent:
    """Read raw text from standard input.

    Args:
        None

    Returns:
        :class:`~shard.pipeline.ExtractedContent` with the stdin content.

    Raises:
        ExtractionError: If stdin is empty.
    """
    text = sys.stdin.read()
    if not text.strip():
        raise ExtractionError("No content received from stdin.")

    # Use the first non-empty line as a provisional title
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    title = first_line[:120] if first_line else "stdin"

    return ExtractedContent(
        text=text,
        source="stdin",
        source_type=SourceType.STDIN,
        title=title,
        metadata={"char_count": str(len(text))},
    )


def _extract_text(text: str) -> ExtractedContent:
    """Treat the input string itself as raw text content.

    Args:
        text: Raw text provided directly.

    Returns:
        :class:`~shard.pipeline.ExtractedContent` wrapping the raw text.

    Raises:
        ExtractionError: If ``text`` is blank.
    """
    if not text.strip():
        raise ExtractionError("Input text is empty.")

    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    title = first_line[:120] if first_line else "text"

    return ExtractedContent(
        text=text,
        source="text",
        source_type=SourceType.TEXT,
        title=title,
        metadata={"char_count": str(len(text))},
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _looks_like_pdf(path: str) -> bool:
    """Return True if *path* refers to a PDF file.

    Checks the extension first (cheap), then the magic bytes if the file
    exists but has a non-``.pdf`` extension.

    Args:
        path: String to test.

    Returns:
        ``True`` when the path points to (or names) a PDF.
    """
    # Avoid treating URLs or oversized strings as file paths.
    if path.startswith(("http://", "https://")) or len(path) > 4096:
        return False

    p = Path(path)

    if p.suffix.lower() == ".pdf":
        return True

    if p.is_file():
        try:
            with p.open("rb") as fh:
                return fh.read(4) == b"%PDF"
        except OSError:
            return False

    return False


def _parse_youtube_video_id(url: str) -> str | None:
    """Extract the YouTube video ID from a URL.

    Handles:
    - ``https://www.youtube.com/watch?v=<ID>``
    - ``https://youtu.be/<ID>``
    - ``https://www.youtube.com/embed/<ID>``
    - ``https://www.youtube.com/shorts/<ID>``

    Args:
        url: A YouTube URL string.

    Returns:
        The 11-character video ID, or ``None`` if it cannot be determined.
    """
    parsed = urlparse(url)

    # youtu.be/<ID>
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/").split("/")[0]
        return video_id or None

    # youtube.com/watch?v=<ID>
    if "youtube.com" in parsed.netloc:
        # /watch?v=
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

        # /embed/<ID>, /shorts/<ID>, /v/<ID>
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) >= 2 and path_parts[0] in ("embed", "shorts", "v", "e"):
            return path_parts[1]

    # Last-resort regex for any YouTube-like URL
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


def _fetch_youtube_title(url: str) -> str | None:
    """Attempt to retrieve a YouTube video's title from the page HTML.

    This is a best-effort operation; failures are silently swallowed so that
    transcript extraction can still proceed with the video ID as a fallback
    title.

    Args:
        url: The YouTube video URL.

    Returns:
        The page ``<title>`` string with the trailing " - YouTube" suffix
        stripped, or ``None`` on any error.
    """
    try:
        import httpx  # type: ignore[import-untyped]
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]

        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(
                url,
                headers={"User-Agent": "shard-cli/0.1 (+https://github.com/shard-cli)"},
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        title_tag = soup.find("title")
        if title_tag:
            raw = title_tag.get_text(strip=True)
            # Strip the " - YouTube" suffix that YouTube always appends
            return re.sub(r"\s*[-–]\s*YouTube\s*$", "", raw).strip() or None
    except Exception:
        pass

    return None
