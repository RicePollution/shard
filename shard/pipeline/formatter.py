"""AI-powered note formatting module.

Transforms raw extracted content into structured, well-formatted markdown
notes by leveraging an LLM to generate titles, tags, summaries, and
clean note bodies.
"""

from __future__ import annotations

import logging
import re

from shard.models import complete
from shard.pipeline import ExtractedContent, FormattedNote, FormattingError

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 24_000
TRUNCATION_NOTICE = "\n\n[Content truncated for processing]"

SYSTEM_PROMPT = """\
You are a knowledge management assistant. Your job is to transform raw \
content into a well-structured note.

You MUST output exactly the following sections in order, each starting on \
its own line:

TITLE: A concise, descriptive title for the note
TAGS: lowercase, comma-separated tags (between 3 and 7 tags)
SUMMARY: A 1-2 sentence summary of the content
BODY:
The full note body in clean markdown. Use headers (##, ###), bullet points, \
bold for key concepts, and code blocks where appropriate. Organize the \
information logically and make it easy to scan and reference later.\
"""


def format_note(extracted: ExtractedContent) -> FormattedNote:
    """Format extracted content into a structured note via an LLM.

    Parameters
    ----------
    extracted:
        The raw extracted content to format.

    Returns
    -------
    FormattedNote
        A fully structured note with title, tags, summary, and markdown body.

    Raises
    ------
    FormattingError
        If the model call fails or the response cannot be parsed.
    """
    text = _truncate(extracted.text)

    source_label = extracted.source_type.name.lower()
    prompt = (
        f"Format the following {source_label} content into a structured note.\n\n"
        f"Source: {extracted.source}\n"
    )
    if extracted.title:
        prompt += f"Original title: {extracted.title}\n"
    prompt += f"\n---\n\n{text}"

    try:
        response = complete(prompt, system=SYSTEM_PROMPT)
    except Exception as exc:
        raise FormattingError(f"Model call failed during formatting: {exc}") from exc

    if not response or not response.strip():
        raise FormattingError("Model returned an empty response")

    parsed = _parse_response(response)

    return FormattedNote(
        title=parsed["title"],
        tags=[t.strip() for t in parsed["tags"].split(",") if t.strip()],
        summary=parsed["summary"],
        body=parsed["body"],
        source=extracted.source,
        source_type=extracted.source_type,
        metadata=extracted.metadata,
    )


def _truncate(text: str) -> str:
    """Truncate text to stay within model context limits.

    If *text* exceeds ``MAX_TEXT_LENGTH`` characters it is cut and a
    truncation notice is appended.
    """
    if len(text) <= MAX_TEXT_LENGTH:
        return text
    return text[:MAX_TEXT_LENGTH] + TRUNCATION_NOTICE


def _parse_response(response: str) -> dict[str, str]:
    """Parse the structured LLM response into its component sections.

    Expects markers ``TITLE:``, ``TAGS:``, ``SUMMARY:``, and ``BODY:`` in the
    response.  Falls back to heuristics when the model does not follow the
    expected format exactly.

    Returns
    -------
    dict with keys ``title``, ``tags``, ``summary``, ``body``.
    """
    # Try structured parsing first.
    title = _extract_field(response, "TITLE")
    tags = _extract_field(response, "TAGS")
    summary = _extract_field(response, "SUMMARY")
    body = _extract_body(response)

    # If we got at least a title and body from the structured format, use them.
    if title and body:
        return {
            "title": title,
            "tags": tags or "note",
            "summary": summary or "",
            "body": body,
        }

    # Fallback: first line as title, everything else as body.
    logger.warning("LLM response did not follow expected format; using fallback parsing")
    lines = response.strip().splitlines()
    if not lines:
        raise FormattingError("Model returned an unparseable empty response")

    fallback_title = lines[0].strip().lstrip("# ").strip()
    fallback_body = "\n".join(lines[1:]).strip() if len(lines) > 1 else lines[0]

    return {
        "title": fallback_title,
        "tags": "note",
        "summary": "",
        "body": fallback_body,
    }


def _extract_field(response: str, field: str) -> str:
    """Extract a single-line field value like ``TITLE: Some Title``.

    The match is case-insensitive and strips surrounding whitespace.
    """
    pattern = rf"^{field}\s*:\s*(.+)$"
    match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def _extract_body(response: str) -> str:
    """Extract everything after the ``BODY:`` marker."""
    pattern = r"(?i)^BODY\s*:\s*\n?"
    match = re.search(pattern, response, re.MULTILINE)
    if match:
        return response[match.end():].strip()
    return ""
