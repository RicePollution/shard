"""ChromaDB indexing for semantic search."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from shard.config import ShardConfig
from shard.pipeline import FormattedNote, IndexedNote, IndexingError, SourceType
from shard.vault import list_shards, parse_frontmatter, read_note

_console = Console(stderr=True)

# ── Text chunking ─────────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split *text* into overlapping word-based chunks.

    The text is tokenised on whitespace.  Chunks of *chunk_size* words are
    produced by sliding a window forward by ``chunk_size - overlap`` words at a
    time so that consecutive chunks share *overlap* words of context.

    An empty or whitespace-only *text* returns an empty list.  A text shorter
    than *chunk_size* words is returned as a single chunk.

    Args:
        text: The source text to split.
        chunk_size: Maximum number of words in each chunk.
        overlap: Number of words shared between consecutive chunks.  Must be
            strictly less than *chunk_size*; if it is not, it is clamped to
            ``chunk_size - 1``.

    Returns:
        A list of chunk strings.  Each string is the space-joined words of one
        window.
    """
    words = text.split()
    if not words:
        return []

    # Guard: overlap must be smaller than chunk_size to avoid an infinite loop.
    overlap = min(overlap, chunk_size - 1)
    step = chunk_size - overlap

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += step

    return chunks


# ── ChromaDB collection ───────────────────────────────────────────────────────


def _get_collection(config: ShardConfig) -> chromadb.Collection:
    """Return (or create) the ``shard_notes`` ChromaDB collection.

    The collection is stored in a :class:`chromadb.PersistentClient` at
    ``config.chroma_path`` and uses a
    :class:`~chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction`
    backed by ``config.embedding_model``.

    Args:
        config: Shard runtime configuration.

    Returns:
        The ``shard_notes`` :class:`chromadb.Collection`.
    """
    client = chromadb.PersistentClient(path=str(config.chroma_path))
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=config.embedding_model)
    return client.get_or_create_collection(
        name="shard_notes",
        embedding_function=embedding_fn,  # type: ignore[arg-type]
    )


# ── Note indexing ─────────────────────────────────────────────────────────────


def index_note(note: FormattedNote, path: Path, config: ShardConfig) -> IndexedNote:
    """Chunk *note* and upsert all chunks into ChromaDB.

    Each chunk receives a stable document ID of the form
    ``<stem>_chunk_<i>`` so that repeated calls are idempotent (upsert
    semantics).

    Args:
        note: The fully-formed note to index.
        path: Vault path of the saved Markdown file; its ``stem`` is used for
            chunk ID generation.
        config: Shard runtime configuration.

    Returns:
        An :class:`~shard.pipeline.IndexedNote` with ``num_chunks`` set to the
        number of chunks that were upserted.

    Raises:
        IndexingError: If ChromaDB raises any error during the upsert.
    """
    try:
        chunks = chunk_text(note.body)

        # ChromaDB requires at least one document in an upsert call; if the
        # body is empty, index a single placeholder so the note is still
        # discoverable by its metadata.
        if not chunks:
            chunks = [note.summary or note.title]

        ids = [f"{path.stem}_chunk_{i}" for i in range(len(chunks))]

        # ChromaDB metadata values must be str, int, float, or bool.
        metadata = {
            "title": note.title,
            "source": note.source,
            "source_type": note.source_type.name.lower(),
            "tags": ", ".join(note.tags),
            "path": str(path),
        }
        # Each chunk shares identical metadata — ChromaDB stores it per document.
        metadatas = [metadata] * len(chunks)

        collection = _get_collection(config)
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)

        return IndexedNote(
            path=path,
            title=note.title,
            tags=note.tags,
            summary=note.summary,
            num_chunks=len(chunks),
        )
    except IndexingError:
        raise
    except Exception as exc:
        raise IndexingError(
            f"Failed to index note '{path}': {exc}"
        ) from exc


# ── Vault reindex ─────────────────────────────────────────────────────────────


def reindex_vault(config: ShardConfig) -> int:
    """Reindex every Shard note in the vault into ChromaDB.

    Iterates over all notes returned by :func:`~shard.vault.list_shards`,
    parses their frontmatter, and upserts their chunks into the
    ``shard_notes`` collection.  Progress is reported on *stderr* via Rich.

    Args:
        config: Shard runtime configuration.

    Returns:
        The total number of chunks upserted across all notes.

    Raises:
        IndexingError: If a note cannot be indexed.  Earlier successfully
            indexed notes are not rolled back.
    """
    note_paths = list_shards(config)
    total_chunks = 0

    if not note_paths:
        _console.print("[yellow]No shard notes found — nothing to index.[/yellow]")
        return 0

    _console.print(
        f"[bold]Reindexing[/bold] {len(note_paths)} note(s) into ChromaDB…"
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=_console,
        transient=True,
    ) as progress:
        task = progress.add_task("Indexing notes", total=len(note_paths))

        for note_path in note_paths:
            progress.update(task, description=f"[cyan]{note_path.name}[/cyan]")

            content = read_note(note_path)
            metadata, body = parse_frontmatter(content)

            # Reconstruct a FormattedNote from the stored frontmatter.
            raw_tags = metadata.get("tags", [])
            tags: list[str] = raw_tags if isinstance(raw_tags, list) else []

            source_type_raw: str = metadata.get("source_type", "text")
            try:
                source_type = SourceType[source_type_raw.upper()]
            except KeyError:
                source_type = SourceType.TEXT

            note = FormattedNote(
                title=metadata.get("title", note_path.stem),
                tags=tags,
                summary=metadata.get("summary", ""),
                body=body,
                source=metadata.get("source", str(note_path)),
                source_type=source_type,
            )

            indexed = index_note(note, note_path, config)
            total_chunks += indexed.num_chunks
            progress.advance(task)

    _console.print(
        f"[green]Done.[/green] Indexed [bold]{len(note_paths)}[/bold] note(s) "
        f"as [bold]{total_chunks}[/bold] chunk(s)."
    )
    return total_chunks
