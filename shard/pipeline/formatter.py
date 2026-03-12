"""AI-powered note formatting module.

Transforms raw extracted content into structured, well-formatted markdown
notes by leveraging an LLM to generate titles, tags, summaries, and
clean note bodies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from collections.abc import Callable
from typing import Any

from shard.models import async_complete, complete
from shard.pipeline import ExtractedContent, FormattedNote, FormattingError

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 24_000
TRUNCATION_NOTICE = "\n\n[Content truncated for processing]"
STYLE_PROFILE_PATH = Path.home() / ".shard" / "style.json"

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
    system, prompt = _build_prompt(extracted, text)

    try:
        response = complete(prompt, system=system)
    except Exception as exc:
        raise FormattingError(f"Model call failed during formatting: {exc}") from exc

    if not response or not response.strip():
        raise FormattingError("Model returned an empty response")

    parsed = _parse_response(response)

    title = _normalize_title(parsed["title"])

    return FormattedNote(
        title=title,
        tags=[t.strip() for t in parsed["tags"].split(",") if t.strip()],
        summary=parsed["summary"],
        body=parsed["body"],
        source=extracted.source,
        source_type=extracted.source_type,
        metadata=extracted.metadata,
    )


def _build_prompt(extracted: ExtractedContent, text: str) -> tuple[str, str]:
    """Build the system prompt and user prompt for note formatting.

    If a style profile exists at ``~/.shard/style.json``, uses a styled
    prompt that matches the user's vault writing patterns.  When the profile
    contains ``real_excerpts``, a strict exemplar-based prompt is used to
    produce notes that are indistinguishable from the user's own writing.
    Otherwise falls back to the generic ``SYSTEM_PROMPT``.

    Returns:
        A ``(system, user_prompt)`` tuple.
    """
    style = _load_style_data()
    source_label = extracted.source_type.name.lower()

    if style is not None:
        real_excerpts: list[str] = style.get("real_excerpts", [])
        forbidden_patterns: list[str] = style.get("forbidden_patterns", [])
        fingerprints: list[str] = style.get("fingerprints", [])
        frontmatter_template: str = style.get("frontmatter_template", "")

        if real_excerpts:
            # Strict exemplar-based prompt for high-fidelity style matching.
            excerpts_block = "\n---\n".join(real_excerpts)

            fingerprint_rules = "\n".join(
                f"{i + 1}. {fp}" for i, fp in enumerate(fingerprints)
            )

            forbidden_block = ""
            if forbidden_patterns:
                forbidden_block = (
                    "\n\nTHINGS YOU MUST NEVER DO:\n"
                    + "\n".join(
                        f"{i + 1}. {fp}"
                        for i, fp in enumerate(forbidden_patterns)
                    )
                )

            system = (
                "You are a note-writing assistant. Your ONLY job is to write notes that are\n"
                "indistinguishable from notes written by this specific person. You must follow\n"
                "their style exactly.\n\n"
                "REAL EXAMPLES of their notes (copy this style exactly):\n"
                f"---\n{excerpts_block}\n---\n\n"
                "RULES YOU MUST FOLLOW (these are not suggestions):\n"
                f"{fingerprint_rules}\n"
                f"{forbidden_block}\n\n"
                "FRONTMATTER: always use exactly this structure:\n"
                f"{frontmatter_template}\n\n"
                "The title field must use plain spaces, never hyphens or underscores.\n"
                "Example: 'The American Revolution' not 'The-American-Revolution'\n\n"
                "If you deviate from any of these rules, you have failed.\n\n"
                "You MUST output exactly the following sections in order, each "
                "starting on its own line:\n\n"
                "TITLE: A concise, descriptive title for the note\n"
                "TAGS: comma-separated tags matching the vault's format\n"
                "SUMMARY: A 1-2 sentence summary of the content\n"
                "BODY:\nThe full note body following the style above."
            )
        else:
            # Legacy style-aware prompt (no real excerpts available).
            fp_block = "\n".join(fingerprints)
            tone = "\n".join(style.get("tone_examples", []))

            system = (
                "You are writing a new note for an Obsidian vault. You must "
                "match the owner's exact writing style — do not deviate.\n\n"
                "STYLE FINGERPRINT (follow these rules exactly):\n"
                f"{fp_block}\n\n"
                "FRONTMATTER TEMPLATE (use this exact structure):\n"
                f"{frontmatter_template}\n\n"
                "NOTE TEMPLATE (follow this skeleton):\n"
                f"{style.get('template', '')}\n\n"
                "STYLE RULES (detailed guide):\n"
                f"{style.get('style_rules', '')}\n\n"
                "TONE EXAMPLES (match this voice exactly):\n"
                f"{tone}\n\n"
                "Now write a new note about the following content.\n"
                "Use the template above as your structure. Fill in real content.\n"
                "Do not add sections that aren't in the template.\n"
                "Do not change the heading names.\n"
                f"Match the tag format exactly: {style.get('tag_format', '')}\n\n"
                "You MUST output exactly the following sections in order, each "
                "starting on its own line:\n\n"
                "TITLE: A concise, descriptive title for the note\n"
                "TAGS: comma-separated tags matching the vault's format\n"
                "SUMMARY: A 1-2 sentence summary of the content\n"
                "BODY:\nThe full note body following the template above."
            )
    else:
        system = SYSTEM_PROMPT

    prompt = f"Format the following {source_label} content into a structured note.\n\n"
    prompt += f"Source: {extracted.source}\n"
    if extracted.title:
        prompt += f"Original title: {extracted.title}\n"
    prompt += f"\n---\n\n{text}"

    return system, prompt


def _normalize_title(title: str) -> str:
    """Normalize a note title to use plain spaces instead of hyphens or underscores.

    Replaces hyphens and underscores with spaces and collapses any runs of
    multiple spaces into a single space.
    """
    title = title.replace("-", " ").replace("_", " ")
    return " ".join(title.split())


def _load_style_data() -> dict | None:
    """Load the style profile JSON from disk if it exists.

    Returns:
        The parsed style data dict, or None if no profile exists.
    """
    if not STYLE_PROFILE_PATH.exists():
        return None

    try:
        return json.loads(STYLE_PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not load style profile from %s", STYLE_PROFILE_PATH)
        return None


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


def _parse_json_response(response: str) -> dict[str, Any]:
    """Extract and parse a JSON object from an LLM response.

    Handles responses where JSON may be wrapped in markdown code fences.
    """
    text = response.strip()
    if text.startswith("```"):
        newline_idx = text.find("\n")
        if newline_idx != -1:
            text = text[newline_idx + 1:]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as exc:
        raise FormattingError(f"Failed to parse JSON from model response: {exc}") from exc


def format_notes(
    extracted: ExtractedContent,
    single: bool = False,
    on_status: Callable[[str], None] | None = None,
) -> list[FormattedNote]:
    """Format extracted content into one or more structured notes.

    Parameters
    ----------
    extracted:
        The raw extracted content to format.
    single:
        When True, bypasses atomic splitting and returns a single note
        (wrapped in a list for a uniform return type).

    Returns
    -------
    list[FormattedNote]
        One FormattedNote when *single* is True, otherwise multiple
        atomic notes plus a parent index note.
    """
    if single:
        return [format_note(extracted)]
    return _format_atomic_notes(extracted, on_status=on_status)


def _format_atomic_notes(
    extracted: ExtractedContent,
    on_status: Callable[[str], None] | None = None,
) -> list[FormattedNote]:
    """Split extracted content into multiple atomic notes via a two-stage LLM process.

    Stage A decomposes the content into distinct subtopics.
    Stage B generates one note per subtopic plus a parent index note.
    All notes are interlinked with [[wikilinks]].
    """
    text = _truncate(extracted.text)
    style = _load_style_data()
    style_injection = _build_style_injection(style)

    # ── Stage A: Topic decomposition ──
    if on_status:
        on_status("Decomposing into subtopics...")
    decomposition = _stage_a_decompose(text, extracted)

    parent_topic = decomposition["parent_topic"]
    parent_summary = decomposition["parent_summary"]
    subtopics = decomposition["subtopics"]

    # Hard enforce subnote count: max 5, min 3 (otherwise single note)
    if len(subtopics) > 5:
        subtopics = subtopics[:5]
    if len(subtopics) < 3:
        return [format_note(extracted)]

    all_titles = [st["title"] for st in subtopics]

    # ── Stage B: Generate one note per subtopic (concurrent) ──
    if on_status:
        on_status(f"Generating {len(subtopics)} notes concurrently...")

    notes = asyncio.run(_generate_subtopics_concurrent(
        subtopics, parent_topic, parent_summary, all_titles,
        style_injection, extracted,
    ))

    # ── Parent index note ──
    if on_status:
        on_status(f"Saving and indexing {len(notes)} notes...")

    parent_note = _generate_parent_index(
        parent_topic, parent_summary, subtopics, notes,
        style_injection, extracted,
    )

    # Parent first, then subtopics
    return [parent_note] + notes


async def _generate_subtopics_concurrent(
    subtopics: list[dict[str, str]],
    parent_topic: str,
    parent_summary: str,
    all_titles: list[str],
    style_injection: str,
    extracted: ExtractedContent,
) -> list[FormattedNote]:
    """Run Stage B note generation concurrently with a semaphore-capped concurrency.

    Up to 3 subtopic notes are generated simultaneously.  Any individual
    failure is retried exactly once before the entire batch is aborted.
    """
    sem = asyncio.Semaphore(3)

    async def generate_one(subtopic: dict[str, str]) -> FormattedNote:
        async with sem:
            return await _stage_b_generate_subtopic_async(
                subtopic, parent_topic, parent_summary, all_titles,
                style_injection, extracted,
            )

    tasks = [generate_one(st) for st in subtopics]
    results: list[FormattedNote | BaseException] = await asyncio.gather(
        *tasks, return_exceptions=True
    )

    final_notes: list[FormattedNote] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            subtopic = subtopics[i]
            logger.warning(
                "Note generation failed for '%s', retrying: %s",
                subtopic["title"], result,
            )
            try:
                result = await _stage_b_generate_subtopic_async(
                    subtopic, parent_topic, parent_summary, all_titles,
                    style_injection, extracted,
                )
            except Exception as exc:
                raise FormattingError(
                    f"Failed to generate note for '{subtopic['title']}' after retry: {exc}"
                ) from exc
        final_notes.append(result)  # type: ignore[arg-type]

    return final_notes


async def _stage_b_generate_subtopic_async(
    subtopic: dict[str, str],
    parent_topic: str,
    parent_summary: str,
    all_titles: list[str],
    style_injection: str,
    extracted: ExtractedContent,
) -> FormattedNote:
    """Async variant of :func:`_stage_b_generate_subtopic` using ``async_complete``."""
    sibling_titles = [t for t in all_titles if t != subtopic["title"]]
    siblings_str = "\n".join(f"- [[{t}]]" for t in sibling_titles)

    system = (
        "You are writing an atomic Obsidian note. An atomic note covers "
        "ONLY one specific concept — nothing else."
    )
    prompt = (
        f"Write an atomic Obsidian note about this specific topic.\n\n"
        f"Topic: {subtopic['title']}\n"
        f"Focus: {subtopic.get('focus', '')}\n"
        f"Source material: {subtopic.get('relevant_section', '')}\n"
        f"Overall source context: {parent_summary}\n\n"
        "All other notes being created from this same source:\n"
        f"{siblings_str}\n\n"
        "Rules:\n"
        "- Cover ONLY the stated topic — nothing else\n"
        "- Length: 100-300 words of actual content\n"
        "- Use [[wikilinks]] to link to the other notes listed above "
        "wherever they are genuinely relevant\n"
        f"- Link back to the parent index note: [[{parent_topic}]]\n"
        "- End with a ## Links section listing all sibling note links\n\n"
        f"{style_injection}\n\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "title": "exact note title",\n'
        '  "slug": "kebab-case-filename",\n'
        '  "tags": ["tag1", "tag2"],\n'
        '  "markdown": "complete note content including any frontmatter"\n'
        "}"
    )

    try:
        response = await async_complete(prompt, system=system)
    except Exception as exc:
        raise FormattingError(
            f"Model call failed generating note for '{subtopic['title']}': {exc}"
        ) from exc

    if not response or not response.strip():
        raise FormattingError(f"Model returned empty response for '{subtopic['title']}'")

    data = _parse_json_response(response)

    title = _normalize_title(data.get("title", subtopic["title"]))

    return FormattedNote(
        title=title,
        tags=data.get("tags", []),
        summary=subtopic.get("focus", ""),
        body=data.get("markdown", ""),
        source=extracted.source,
        source_type=extracted.source_type,
        metadata=extracted.metadata,
    )


def _build_style_injection(style: dict[str, Any] | None) -> str:
    """Build a style injection string from the style profile, if available.

    When the profile contains ``real_excerpts``, a strict exemplar-based
    injection is produced.  Otherwise falls back to the legacy fingerprint
    injection for backward compatibility with older style profiles.
    """
    if style is None:
        return ""

    real_excerpts: list[str] = style.get("real_excerpts", [])
    forbidden_patterns: list[str] = style.get("forbidden_patterns", [])
    fingerprints: list[str] = style.get("fingerprints", [])

    if real_excerpts:
        excerpts_block = "\n---\n".join(real_excerpts)

        fingerprint_rules = "\n".join(
            f"{i + 1}. {fp}" for i, fp in enumerate(fingerprints)
        )

        forbidden_block = ""
        if forbidden_patterns:
            forbidden_block = (
                "\nTHINGS YOU MUST NEVER DO:\n"
                + "\n".join(
                    f"{i + 1}. {fp}" for i, fp in enumerate(forbidden_patterns)
                )
                + "\n"
            )

        return (
            "\nYour ONLY job is to write notes that are indistinguishable from "
            "notes written by this specific person.\n\n"
            "REAL EXAMPLES of their notes (copy this style exactly):\n"
            f"---\n{excerpts_block}\n---\n\n"
            "RULES YOU MUST FOLLOW (these are not suggestions):\n"
            f"{fingerprint_rules}\n\n"
            f"{forbidden_block}"
            f"FRONTMATTER: always use exactly this structure:\n"
            f"{style.get('frontmatter_template', '')}\n\n"
            "The title field must use plain spaces, never hyphens or underscores.\n"
            "Example: 'The American Revolution' not 'The-American-Revolution'\n\n"
            f"TAG FORMAT: {style.get('tag_format', '')}\n"
            "If you deviate from any of these rules, you have failed.\n"
        )

    # Legacy fallback — no real excerpts available.
    fp_block = "\n".join(fingerprints)
    return (
        "\nMatch the user's writing style exactly:\n"
        f"STYLE FINGERPRINT:\n{fp_block}\n"
        f"TAG FORMAT: {style.get('tag_format', '')}\n"
        f"TEMPLATE:\n{style.get('template', '')}\n"
    )


def _stage_a_decompose(text: str, extracted: ExtractedContent) -> dict[str, Any]:
    """Stage A: Identify distinct subtopics in the content."""
    system = (
        "You are building an Obsidian knowledge vault using atomic notes. "
        "Atomic notes follow one strict rule: one note = one idea."
    )
    prompt = (
        "Read this content carefully:\n\n"
        f"{text}\n\n"
        "Identify every distinct concept, subtopic, or idea in this content "
        "that deserves its own dedicated note. Think of these as the "
        "fundamental building blocks.\n\n"
        "Rules for splitting:\n"
        "- Each topic must be genuinely distinct and self-contained\n"
        "- You MUST return between 3 and 5 subtopics. No more, no fewer.\n"
        "- If the content truly only has 1-2 distinct concepts, return exactly 2\n"
        "- If the content has more than 5 distinct concepts, pick the 5 most important\n"
        "- Do not create a topic for introductions, conclusions, or meta info\n"
        "- Each topic should be specific enough to fill 100-300 words\n"
        "- The topics together should cover the entire source completely\n\n"
        "Also identify one PARENT topic — the overarching subject that all "
        "subtopics belong to. This becomes the index note.\n\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "parent_topic": "the overarching subject title",\n'
        '  "parent_summary": "2-3 sentence summary of the whole source",\n'
        '  "subtopics": [\n'
        "    {\n"
        '      "title": "specific concept name",\n'
        '      "focus": "one sentence describing exactly what this note covers",\n'
        '      "relevant_section": "brief quote or description of source section"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    try:
        response = complete(prompt, system=system)
    except Exception as exc:
        raise FormattingError(f"Model call failed during topic decomposition: {exc}") from exc

    if not response or not response.strip():
        raise FormattingError("Model returned an empty response during topic decomposition")

    data = _parse_json_response(response)

    # Validate required fields
    if "parent_topic" not in data or "subtopics" not in data:
        raise FormattingError("Topic decomposition missing required fields (parent_topic, subtopics)")

    if "parent_summary" not in data:
        raise FormattingError("Topic decomposition missing required field: parent_summary")

    if not isinstance(data.get("subtopics"), list):
        raise FormattingError("Topic decomposition 'subtopics' must be a list")

    for i, st in enumerate(data["subtopics"]):
        if not isinstance(st, dict) or "title" not in st:
            raise FormattingError(
                f"Subtopic {i} missing required 'title' key"
            )

    return data


def _generate_parent_index(
    parent_topic: str,
    parent_summary: str,
    subtopics: list[dict[str, str]],
    child_notes: list[FormattedNote],
    style_injection: str,
    extracted: ExtractedContent,
) -> FormattedNote:
    """Generate the parent index (MOC) note linking to all child notes."""
    children_str = "\n".join(
        f"- [[{note.title}]]: {st.get('focus', '')}"
        for st, note in zip(subtopics, child_notes)
    )

    system = (
        "You are writing a parent index note (Map of Content) for an "
        "Obsidian vault. This note links to all child notes from the same source."
    )
    prompt = (
        "Write a parent index note that serves as the entry point for a "
        "collection of atomic notes all derived from the same source.\n\n"
        f"Parent topic: {parent_topic}\n"
        f"Summary: {parent_summary}\n"
        f"Source: {extracted.source}\n\n"
        f"Child notes:\n{children_str}\n\n"
        "This note should:\n"
        "- Open with a 2-3 sentence summary of the source\n"
        "- Have a ## Notes section with a [[wikilink]] and one-line "
        "description for every child note\n"
        "- Have a ## Source section with the original URL or filename\n"
        "- Be clearly an index/MOC (map of content) note\n"
        "- Use the tag #index or #moc\n\n"
        f"{style_injection}\n\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "title": "parent note title",\n'
        '  "slug": "kebab-case-filename",\n'
        '  "tags": ["index", "tag1"],\n'
        '  "markdown": "complete note content"\n'
        "}"
    )

    try:
        response = complete(prompt, system=system)
    except Exception as exc:
        raise FormattingError(
            f"Model call failed generating parent index note: {exc}"
        ) from exc

    if not response or not response.strip():
        raise FormattingError("Model returned empty response for parent index note")

    data = _parse_json_response(response)

    tags = data.get("tags", ["index"])
    if "index" not in tags and "moc" not in tags:
        tags.insert(0, "index")

    return FormattedNote(
        title=data.get("title", parent_topic),
        tags=tags,
        summary=parent_summary,
        body=data.get("markdown", ""),
        source=extracted.source,
        source_type=extracted.source_type,
        metadata=extracted.metadata,
    )
