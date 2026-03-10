"""Obsidian vault file I/O."""

from __future__ import annotations

import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from shard.config import ShardConfig
from shard.pipeline import FormattedNote, VaultError

# Relative path inside the vault where Shard notes are written.
_SHARDS_SUBDIR = Path("Imported") / "Shards"


# ── Slug ─────────────────────────────────────────────────────────────────────


def slugify(text: str) -> str:
    """Convert arbitrary text into a filesystem- and URL-safe slug.

    Args:
        text: The source string to slugify (e.g. a note title).

    Returns:
        A lowercase, hyphen-separated slug truncated to 80 characters.

    Examples:
        >>> slugify("Hello, World!")
        'hello-world'
        >>> slugify("  Multiple   Spaces  ")
        'multiple-spaces'
        >>> slugify("A" * 100)
        'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    """
    slug = text.lower()
    # Replace every run of non-alphanumeric characters with a single hyphen.
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    # Strip leading / trailing hyphens that may have been introduced above.
    slug = slug.strip("-")
    # Hard-truncate at 80 characters, then strip any newly dangling hyphen.
    slug = slug[:80].rstrip("-")
    return slug


# ── Frontmatter ───────────────────────────────────────────────────────────────


def _build_frontmatter(note: FormattedNote) -> str:
    """Render YAML frontmatter for *note* without an external YAML library.

    The produced block is intentionally minimal:

    * Scalar values are written as bare ``key: value`` pairs.
    * ``tags`` is serialised as a YAML block sequence.
    * All values are single-quoted to guard against special characters; single
      quotes inside values are escaped by doubling them (``'`` → ``''``).

    Args:
        note: The formatted note whose metadata should be serialised.

    Returns:
        A string starting and ending with ``---\\n``.
    """

    def _quote(value: str) -> str:
        """Wrap *value* in single quotes, escaping any interior single quotes."""
        return "'" + value.replace("'", "''") + "'"

    date_iso = datetime.now(tz=timezone.utc).date().isoformat()
    source_type_value = note.source_type.name.lower()

    lines: list[str] = ["---"]
    lines.append(f"title: {_quote(note.title)}")

    # Block-sequence tags — always emit the key even when the list is empty.
    if note.tags:
        lines.append("tags:")
        for tag in note.tags:
            lines.append(f"  - {_quote(tag)}")
    else:
        lines.append("tags: []")

    lines.append(f"source: {_quote(note.source)}")
    lines.append(f"source_type: {_quote(source_type_value)}")
    lines.append(f"date: {_quote(date_iso)}")
    lines.append(f"summary: {_quote(note.summary)}")
    lines.append("---")

    return "\n".join(lines) + "\n"


# ── Core vault operations ─────────────────────────────────────────────────────


def save_note(note: FormattedNote, config: ShardConfig) -> Path:
    """Persist *note* to the vault as a Markdown file with YAML frontmatter.

    The file is written to ``<vault>/Imported/Shards/<slug>.md``.  Parent
    directories are created automatically.  If a file at the target path
    already exists it is **overwritten** — callers that need collision-safe
    writes should check first via :func:`list_shards`.

    Args:
        note: The fully-formed note to serialise.
        config: Shard runtime configuration supplying the vault root.

    Returns:
        The absolute :class:`~pathlib.Path` of the written file.

    Raises:
        VaultError: If the file cannot be written (e.g. permission denied).
    """
    slug = slugify(note.title) or "untitled"
    dest_dir = config.vault_path / _SHARDS_SUBDIR
    dest_path = dest_dir / f"{slug}.md"

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VaultError(f"Cannot create shard directory '{dest_dir}': {exc}") from exc

    frontmatter = _build_frontmatter(note)
    # Separate frontmatter from body with a blank line for readability.
    content = frontmatter + "\n" + note.body

    try:
        dest_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise VaultError(f"Cannot write note to '{dest_path}': {exc}") from exc

    return dest_path


def walk_vault(config: ShardConfig) -> list[Path]:
    """Return all Markdown files found anywhere under the vault root.

    The search is recursive.  Files are returned in the order that
    :meth:`pathlib.Path.rglob` yields them (depth-first, lexicographic within
    each directory).

    Args:
        config: Shard runtime configuration supplying the vault root.

    Returns:
        A list of absolute :class:`~pathlib.Path` objects for every ``*.md``
        file under ``config.vault_path``.

    Raises:
        VaultError: If the vault root does not exist or is not a directory.
    """
    vault = config.vault_path
    if not vault.exists():
        raise VaultError(f"Vault path does not exist: '{vault}'")
    if not vault.is_dir():
        raise VaultError(f"Vault path is not a directory: '{vault}'")

    return sorted(vault.rglob("*.md"))


def list_shards(config: ShardConfig) -> list[Path]:
    """Return all Markdown files inside the Shard import directory.

    Unlike :func:`walk_vault`, this only looks in
    ``<vault>/Imported/Shards/``.  If that sub-directory does not yet exist
    the function returns an empty list rather than raising.

    Args:
        config: Shard runtime configuration supplying the vault root.

    Returns:
        A sorted list of ``*.md`` paths under the Shards directory.
    """
    shards_dir = config.vault_path / _SHARDS_SUBDIR
    if not shards_dir.exists():
        return []
    return sorted(shards_dir.glob("*.md"))


def read_note(path: Path) -> str:
    """Read and return the full text content of a Markdown note.

    Args:
        path: Absolute path to the Markdown file.

    Returns:
        The UTF-8 decoded file content as a plain string.

    Raises:
        VaultError: If the file does not exist or cannot be read.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise VaultError(f"Note not found: '{path}'") from exc
    except OSError as exc:
        raise VaultError(f"Cannot read note '{path}': {exc}") from exc


# ── Frontmatter parser ────────────────────────────────────────────────────────


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse the YAML frontmatter block at the top of *content*.

    The parser is intentionally hand-rolled to avoid a PyYAML dependency.  It
    handles the subset of YAML that :func:`save_note` produces:

    * Simple ``key: value`` pairs — values are returned as strings with any
      surrounding single quotes (and escaped ``''`` sequences) resolved.
    * Block-sequence lists: a key whose value is immediately followed by lines
      beginning with ``  - `` is collected into a Python :class:`list`.
    * The ``tags: []`` inline-empty-list form.
    * Lines that do not match the expected patterns are silently skipped.

    Multiline values and nested mappings are **not** supported.

    Args:
        content: Raw file text, potentially beginning with ``---\\n``.

    Returns:
        A two-tuple of:

        * ``metadata`` — a :class:`dict` mapping frontmatter keys to their
          parsed values (``str`` or ``list[str]``).
        * ``body`` — the remainder of the document after the closing ``---``
          delimiter (leading newline stripped).

    """
    metadata: dict[str, Any] = {}
    body = content

    if not content.startswith("---"):
        return metadata, body

    # Locate the closing delimiter.  We skip the opening "---" line itself.
    first_newline = content.index("\n")
    rest = content[first_newline + 1 :]
    close_pos = rest.find("\n---")
    if close_pos == -1:
        # Malformed frontmatter — treat the whole file as body.
        return metadata, body

    fm_text = rest[:close_pos]
    body = rest[close_pos + 4 :]  # skip "\n---"
    # Strip the single leading newline that separates frontmatter from body.
    if body.startswith("\n"):
        body = body[1:]

    _parse_fm_block(fm_text, metadata)
    return metadata, body


def _parse_fm_block(fm_text: str, metadata: dict[str, Any]) -> None:
    """Populate *metadata* by parsing the raw frontmatter text.

    Mutates *metadata* in place.  Helper for :func:`parse_frontmatter`.

    Args:
        fm_text: The raw text between the opening and closing ``---`` lines.
        metadata: The dict to populate with parsed key/value pairs.
    """
    lines = fm_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        # Skip blank lines.
        if not line.strip():
            i += 1
            continue

        # Top-level key: value pair.
        if ":" in line and not line.startswith(" ") and not line.startswith("-"):
            key, _, raw_value = line.partition(":")
            key = key.strip()
            raw_value = raw_value.strip()

            # Inline empty list: ``tags: []``
            if raw_value == "[]":
                metadata[key] = []
                i += 1
                continue

            # If the value is absent, peek ahead for a block sequence.
            if raw_value == "":
                items: list[str] = []
                i += 1
                while i < len(lines) and lines[i].startswith("  - "):
                    item_raw = lines[i][4:].strip()
                    items.append(_unquote(item_raw))
                    i += 1
                metadata[key] = items
                continue

            metadata[key] = _unquote(raw_value)

        i += 1


def _unquote(value: str) -> str:
    """Strip surrounding single quotes and unescape doubled single quotes.

    If the value is not single-quoted it is returned unchanged.

    Args:
        value: A raw YAML scalar value, possibly single-quoted.

    Returns:
        The unquoted and unescaped string.

    Examples:
        >>> _unquote("'hello'")
        'hello'
        >>> _unquote("'it''s fine'")
        "it's fine"
        >>> _unquote("bare")
        'bare'
    """
    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        return value[1:-1].replace("''", "'")
    return value


# ── Obsidian URI launcher ─────────────────────────────────────────────────────


def open_in_obsidian(path: Path, config: ShardConfig) -> None:
    """Open *path* in the Obsidian desktop application via its URI scheme.

    The ``obsidian://open`` URI requires two query parameters:

    * ``vault`` — the vault *name* (i.e. the final component of the vault
      root path, which Obsidian uses as the vault identifier).
    * ``file`` — the vault-relative path to the note, without the ``.md``
      extension, using forward slashes.

    The URI is dispatched via the OS default handler:

    * Linux: ``xdg-open``
    * macOS: ``open``
    * Other platforms raise :exc:`VaultError`.

    Args:
        path: Absolute path to the ``.md`` file to open.
        config: Shard runtime configuration supplying the vault root.

    Raises:
        VaultError: If the platform is unsupported, the path is not inside the
            vault, or the subprocess fails.
    """
    # Derive the vault-relative path (no extension) for the ``file`` param.
    try:
        rel = path.relative_to(config.vault_path)
    except ValueError as exc:
        raise VaultError(
            f"Path '{path}' is not inside vault '{config.vault_path}'"
        ) from exc

    # Remove .md suffix for the Obsidian URI — it does not expect it.
    file_param = rel.with_suffix("").as_posix()
    vault_name = config.vault_path.name

    uri = (
        f"obsidian://open"
        f"?vault={quote(vault_name, safe='')}"
        f"&file={quote(file_param, safe='/')}"
    )

    system = platform.system()
    if system == "Linux":
        opener = "xdg-open"
    elif system == "Darwin":
        opener = "open"
    else:
        raise VaultError(
            f"Unsupported platform '{system}': cannot open Obsidian URI automatically."
        )

    try:
        subprocess.run(
            [opener, uri],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise VaultError(
            f"'{opener}' not found — cannot open Obsidian URI: {uri}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip()
        raise VaultError(
            f"'{opener}' exited with code {exc.returncode} for URI '{uri}': {stderr}"
        ) from exc
