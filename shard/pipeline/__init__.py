"""Pipeline dataclasses and error hierarchy for Shard."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

# ── Error hierarchy ──────────────────────────────────────────────────────────


class ShardError(Exception):
    """Base error for all Shard operations."""


class ExtractionError(ShardError):
    """Failed to extract content from a source."""


class FormattingError(ShardError):
    """Failed to format extracted content into a note."""


class IndexingError(ShardError):
    """Failed to index a note in ChromaDB."""


class ConfigError(ShardError):
    """Configuration is missing or invalid."""


class ModelError(ShardError):
    """Model invocation failed."""


class VaultError(ShardError):
    """Vault I/O error."""


class LearnError(ShardError):
    """Failed to learn writing style from vault notes."""


class LinkError(ShardError):
    """Failed to generate or apply cross-note links."""


# ── Pipeline dataclasses ─────────────────────────────────────────────────────


class SourceType(Enum):
    PDF = auto()
    URL = auto()
    YOUTUBE = auto()
    STDIN = auto()
    TEXT = auto()


@dataclass
class ExtractedContent:
    text: str
    source: str
    source_type: SourceType
    title: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class FormattedNote:
    title: str
    tags: list[str]
    summary: str
    body: str
    source: str
    source_type: SourceType
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class IndexedNote:
    path: Path
    title: str
    tags: list[str]
    summary: str
    num_chunks: int = 0
