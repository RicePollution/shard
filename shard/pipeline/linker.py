"""Cross-note linking pipeline for building Obsidian knowledge graphs."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from shard.models import complete
from shard.pipeline import LinkError

logger = logging.getLogger(__name__)


@dataclass
class LinkSuggestion:
    """A single link substitution to apply in a note."""

    original_text: str
    linked_text: str
    note_title: str


class Linker:
    """Finds cross-note link opportunities using LLM analysis."""

    def find_links(self, note: str, all_titles: list[str]) -> list[LinkSuggestion]:
        """Find places in a note where other vault titles are meaningfully referenced.

        Args:
            note: The full markdown content of the note to analyze.
            all_titles: All note titles in the vault (for linking targets).

        Returns:
            A list of LinkSuggestion instances for suggested substitutions.

        Raises:
            LinkError: If the model call or response parsing fails.
        """
        if not all_titles:
            return []

        titles_str = "\n".join(all_titles)

        prompt = (
            "You are helping build an Obsidian knowledge graph.\n\n"
            f"Current note:\n{note}\n\n"
            f"Available note titles in this vault:\n{titles_str}\n\n"
            "Find places in the current note where one of the available "
            "titles is meaningfully referenced — either by exact name or "
            "clear concept match. Only suggest links that add genuine "
            "value. Do not link common words, dates, or words under 4 "
            "characters. Do not suggest links for text already inside "
            "[[brackets]].\n\n"
            "Return ONLY a JSON array:\n"
            "[\n"
            "  {\n"
            '    "original_text": "exact text in note to replace",\n'
            '    "linked_text": "[[NoteTitle|display text]]",\n'
            '    "note_title": "the note being linked to"\n'
            "  }\n"
            "]"
        )

        try:
            response = complete(prompt)
        except Exception as exc:
            raise LinkError(f"Model call failed during link analysis: {exc}") from exc

        suggestions = _parse_link_response(response)
        return suggestions


def apply_links(content: str, suggestions: list[LinkSuggestion]) -> str:
    """Apply link substitutions to note content.

    Only replaces the first occurrence of each term per note.
    Never modifies text already inside [[]] or []() links.

    Args:
        content: The original note content.
        suggestions: Link substitutions to apply.

    Returns:
        The content with wikilinks applied.
    """
    for suggestion in suggestions:
        original = suggestion.original_text
        replacement = suggestion.linked_text

        # Skip if the original text is not found
        if original not in content:
            continue

        # Build a regex that matches the original text but NOT inside
        # existing [[...]] or [...](...)
        # Strategy: find the first occurrence that's not inside brackets
        content = _replace_first_outside_links(content, original, replacement)

    return content


def _replace_first_outside_links(content: str, original: str, replacement: str) -> str:
    """Replace the first occurrence of original that is NOT inside [[ ]] or [ ]( ).

    Args:
        content: Full note text.
        original: Text to find and replace.
        replacement: Replacement wikilink text.

    Returns:
        Content with at most one substitution applied.
    """
    # Find all regions that are inside existing links
    # Matches [[...]] and [...](...)
    link_pattern = re.compile(r'\[\[[^\]]*\]\]|\[[^\]]*\]\([^)]*\)')
    protected_ranges: list[tuple[int, int]] = []
    for m in link_pattern.finditer(content):
        protected_ranges.append((m.start(), m.end()))

    # Find all occurrences of the original text
    start = 0
    while True:
        idx = content.find(original, start)
        if idx == -1:
            break

        end_idx = idx + len(original)

        # Check if this occurrence overlaps with any protected range
        overlaps = any(
            not (end_idx <= pr_start or idx >= pr_end)
            for pr_start, pr_end in protected_ranges
        )

        if not overlaps:
            # Safe to replace — do it and return
            return content[:idx] + replacement + content[end_idx:]

        start = end_idx

    return content


def _parse_link_response(response: str) -> list[LinkSuggestion]:
    """Parse the LLM response into a list of LinkSuggestion.

    Handles JSON wrapped in markdown code fences.

    Args:
        response: Raw LLM response text.

    Returns:
        List of parsed LinkSuggestion instances.

    Raises:
        LinkError: If the response cannot be parsed as JSON.
    """
    text = response.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        newline_idx = text.find("\n")
        if newline_idx != -1:
            text = text[newline_idx + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

    # Find JSON array boundaries
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        data: list[dict[str, Any]] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LinkError(f"Failed to parse link suggestions JSON: {exc}") from exc

    suggestions: list[LinkSuggestion] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        original = item.get("original_text", "")
        linked = item.get("linked_text", "")
        title = item.get("note_title", "")
        if original and linked and title:
            suggestions.append(LinkSuggestion(
                original_text=original,
                linked_text=linked,
                note_title=title,
            ))

    return suggestions
