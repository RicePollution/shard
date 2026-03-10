"""Style learning pipeline for analyzing vault note patterns."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shard.models import complete
from shard.pipeline import LearnError

logger = logging.getLogger(__name__)

MAX_SAMPLE_SIZE = 20
QUICK_SAMPLE_SIZE = 5


@dataclass
class StyleProfile:
    """Learned writing style from vault analysis."""

    style_rules: str
    template: str
    fingerprints: list[str]
    frontmatter_template: str
    heading_order: list[str]
    tag_format: str
    avg_word_count: int
    tone_examples: list[str]
    analyzed_at: str
    notes_sampled: int


class Learner:
    """Analyzes vault notes to extract the user's writing style."""

    def analyze(self, notes: list[str], depth: str = "normal") -> StyleProfile:
        """Run style analysis on vault notes.

        Args:
            notes: List of raw markdown note contents from the vault.
            depth: Analysis depth — "quick", "normal", or "deep".

        Returns:
            A StyleProfile capturing the user's writing patterns.

        Raises:
            LearnError: If fewer than 5 notes provided, or LLM calls fail.
        """
        if len(notes) < 5:
            raise LearnError(
                f"Need at least 5 notes to learn style (found {len(notes)}). "
                "Add more notes first, then re-run shard learn."
            )

        if depth == "quick":
            # Sample 5 notes, skip Pass 1, single Pass 2 call
            if len(notes) > QUICK_SAMPLE_SIZE:
                sampled = random.sample(notes, QUICK_SAMPLE_SIZE)
            else:
                sampled = list(notes)
            profile = self._quick_synthesize(sampled)
            return profile

        if depth == "deep":
            # Use ALL notes — no sampling cap
            sampled = list(notes)
        else:
            # Normal: sample up to MAX_SAMPLE_SIZE
            if len(notes) > MAX_SAMPLE_SIZE:
                sampled = random.sample(notes, MAX_SAMPLE_SIZE)
            else:
                sampled = list(notes)

        # PASS 1: Per-note structural extraction
        pass1_results = self._pass1_extract(sampled)

        # PASS 2: Cross-note synthesis
        profile = self._pass2_synthesize(pass1_results, len(sampled))

        return profile

    def _pass1_extract(self, notes: list[str]) -> list[dict[str, Any]]:
        """Extract structural facts from each individual note.

        Args:
            notes: Sampled note contents to analyze individually.

        Returns:
            List of JSON-parsed structural analysis dicts.

        Raises:
            LearnError: If model calls fail.
        """
        results: list[dict[str, Any]] = []

        for note in notes:
            prompt = (
                "Analyze this Obsidian note and extract the following exact "
                "details. Be precise and literal — copy actual examples from "
                "the note, do not describe them.\n\n"
                f"Note content:\n{note}\n\n"
                "Return ONLY a JSON object:\n"
                "{\n"
                "  \"title_format\": \"exact title of this note\",\n"
                "  \"heading_levels_used\": [\"list of heading levels e.g. ##, ###\"],\n"
                "  \"first_section_heading\": \"exact text of first heading if any\",\n"
                "  \"bullet_style\": \"dash / asterisk / none — what they use\",\n"
                "  \"has_frontmatter\": true,\n"
                "  \"frontmatter_fields\": [\"exact field names found e.g. tags, date\"],\n"
                "  \"tag_format\": \"how tags look e.g. #lowercase, [[Category/Tag]]\",\n"
                "  \"tag_count\": 0,\n"
                "  \"tag_examples\": [\"up to 3 actual tags from this note\"],\n"
                "  \"avg_section_length\": \"short(1-3 lines) / medium(4-8) / long(9+)\",\n"
                "  \"uses_bold\": true,\n"
                "  \"uses_italic\": true,\n"
                "  \"uses_callouts\": false,\n"
                "  \"callout_examples\": [\"exact callout types used e.g. > [!note]\"],\n"
                "  \"uses_code_blocks\": false,\n"
                "  \"link_style\": \"wikilink [[]] / markdown []() / both / none\",\n"
                "  \"opens_with\": \"heading / paragraph / bullet / frontmatter\",\n"
                "  \"closes_with\": \"tags / related links / nothing / summary\",\n"
                "  \"word_count\": 0,\n"
                "  \"sentence_examples\": [\"copy 2-3 actual sentences verbatim\"]\n"
                "}"
            )

            try:
                response = complete(prompt)
            except Exception as exc:
                raise LearnError(f"Model call failed during style analysis: {exc}") from exc

            parsed = _parse_json_response(response)
            results.append(parsed)

        return results

    def _quick_synthesize(self, notes: list[str]) -> StyleProfile:
        """Single-pass style synthesis for quick depth mode.

        Skips Pass 1 and sends all notes directly to a synthesis prompt.

        Args:
            notes: Sampled note contents (typically 5).

        Returns:
            A StyleProfile from direct synthesis.

        Raises:
            LearnError: If model call or JSON parsing fails.
        """
        notes_block = "\n\n---\n\n".join(
            f"NOTE {i+1}:\n{note[:2000]}" for i, note in enumerate(notes)
        )

        prompt = (
            f"You are analyzing {len(notes)} complete notes from a personal "
            "Obsidian vault. Rather than per-note structural analysis, "
            "synthesize the writing style directly from these examples.\n\n"
            f"{notes_block}\n\n"
            "Synthesize into:\n\n"
            "1. A STYLE RULES document — concrete rules a writer must follow "
            "to match this vault. Be specific.\n\n"
            "2. A TEMPLATE — a complete blank note skeleton in their exact style.\n\n"
            "3. A STYLE FINGERPRINT — 5-8 one-line rules that are the most "
            "distinctive things about how this person writes.\n\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "style_rules": "full markdown style rules document",\n'
            '  "template": "complete blank note template in their format",\n'
            '  "fingerprints": ["list of 5-8 specific one-line rules"],\n'
            '  "frontmatter_template": "exact frontmatter block to use",\n'
            '  "heading_order": ["typical heading sequence if consistent"],\n'
            '  "tag_format": "exact format e.g. #lowercase-hyphen",\n'
            '  "avg_word_count": 0,\n'
            '  "tone_examples": ["2-3 verbatim sentence examples from notes"]\n'
            "}"
        )

        try:
            response = complete(prompt)
        except Exception as exc:
            raise LearnError(f"Model call failed during quick synthesis: {exc}") from exc

        data = _parse_json_response(response)

        return StyleProfile(
            style_rules=data.get("style_rules", ""),
            template=data.get("template", ""),
            fingerprints=data.get("fingerprints", []),
            frontmatter_template=data.get("frontmatter_template", ""),
            heading_order=data.get("heading_order", []),
            tag_format=data.get("tag_format", ""),
            avg_word_count=int(data.get("avg_word_count", 0)),
            tone_examples=data.get("tone_examples", []),
            analyzed_at=datetime.now(tz=timezone.utc).isoformat(),
            notes_sampled=len(notes),
        )

    def _pass2_synthesize(
        self, pass1_results: list[dict[str, Any]], notes_sampled: int
    ) -> StyleProfile:
        """Synthesize individual note analyses into a unified style profile.

        Args:
            pass1_results: Collected structural analyses from Pass 1.
            notes_sampled: Number of notes that were sampled.

        Returns:
            A complete StyleProfile.

        Raises:
            LearnError: If model call or JSON parsing fails.
        """
        prompt = (
            f"You have analyzed {notes_sampled} notes from a personal Obsidian vault.\n"
            "Here are the structural findings from each note:\n\n"
            f"{json.dumps(pass1_results, indent=2)}\n\n"
            "Now synthesize this into:\n\n"
            "1. A STYLE RULES document — concrete rules a writer must follow "
            "to match this vault. Be specific. Instead of 'uses headers' "
            "say 'always starts with a ## Overview header'. Instead of "
            "'casual tone' copy actual sentence patterns like "
            "'writes in second person, uses em dashes, short sentences'. "
            "Include exact tag format, exact frontmatter fields used, "
            "exact heading patterns.\n\n"
            "2. A TEMPLATE — a complete blank note skeleton in their exact "
            "style with placeholder sections. This should be a real "
            "markdown template someone could fill in. Use their actual "
            "heading names if consistent, their actual frontmatter fields, "
            "their actual tag format.\n\n"
            "3. A STYLE FINGERPRINT — 5-8 one-line rules that are the most "
            "distinctive things about how this person writes. These should "
            "be so specific that if you followed only these rules, a new "
            "note would feel native to the vault.\n"
            "Example good fingerprints:\n"
            "  - Always opens with a one-sentence TL;DR in bold\n"
            "  - Uses ## TL;DR, ## Notes, ## Links as standard headings\n"
            "  - Tags are always lowercase with hyphens: #machine-learning\n"
            "  - Ends every note with a ## Related section\n"
            "Example bad fingerprints (too vague):\n"
            "  - Uses markdown formatting\n"
            "  - Has a casual tone\n\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            "  \"style_rules\": \"full markdown style rules document\",\n"
            "  \"template\": \"complete blank note template in their format\",\n"
            "  \"fingerprints\": [\"list of 5-8 specific one-line rules\"],\n"
            "  \"frontmatter_template\": \"exact frontmatter block to use\",\n"
            "  \"heading_order\": [\"typical heading sequence if consistent\"],\n"
            "  \"tag_format\": \"exact format e.g. #lowercase-hyphen\",\n"
            "  \"avg_word_count\": 0,\n"
            "  \"tone_examples\": [\"2-3 verbatim sentence examples from notes\"]\n"
            "}"
        )

        try:
            response = complete(prompt)
        except Exception as exc:
            raise LearnError(f"Model call failed during style synthesis: {exc}") from exc

        data = _parse_json_response(response)

        return StyleProfile(
            style_rules=data.get("style_rules", ""),
            template=data.get("template", ""),
            fingerprints=data.get("fingerprints", []),
            frontmatter_template=data.get("frontmatter_template", ""),
            heading_order=data.get("heading_order", []),
            tag_format=data.get("tag_format", ""),
            avg_word_count=int(data.get("avg_word_count", 0)),
            tone_examples=data.get("tone_examples", []),
            analyzed_at=datetime.now(tz=timezone.utc).isoformat(),
            notes_sampled=notes_sampled,
        )


def _parse_json_response(response: str) -> dict[str, Any]:
    """Extract and parse a JSON object from an LLM response.

    Handles responses where the JSON may be wrapped in markdown code fences.

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        LearnError: If no valid JSON can be extracted.
    """
    text = response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        newline_idx = text.find("\n")
        if newline_idx != -1:
            text = text[newline_idx + 1:]
            # Remove closing fence
            if text.endswith("```"):
                text = text[:-3].strip()

    # Try to find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LearnError(f"Failed to parse style analysis JSON: {exc}") from exc


def save_style_profile(profile: StyleProfile, path: Path) -> None:
    """Save a StyleProfile to a JSON file.

    Args:
        profile: The style profile to persist.
        path: Destination file path.

    Raises:
        LearnError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(asdict(profile), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        raise LearnError(f"Cannot save style profile to '{path}': {exc}") from exc


def load_style_profile(path: Path) -> StyleProfile | None:
    """Load a StyleProfile from a JSON file.

    Args:
        path: Path to the style profile JSON.

    Returns:
        The loaded StyleProfile, or None if the file does not exist.

    Raises:
        LearnError: If the file exists but cannot be parsed.
    """
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LearnError(f"Cannot read style profile from '{path}': {exc}") from exc

    return StyleProfile(
        style_rules=data.get("style_rules", ""),
        template=data.get("template", ""),
        fingerprints=data.get("fingerprints", []),
        frontmatter_template=data.get("frontmatter_template", ""),
        heading_order=data.get("heading_order", []),
        tag_format=data.get("tag_format", ""),
        avg_word_count=int(data.get("avg_word_count", 0)),
        tone_examples=data.get("tone_examples", []),
        analyzed_at=data.get("analyzed_at", ""),
        notes_sampled=int(data.get("notes_sampled", 0)),
    )
