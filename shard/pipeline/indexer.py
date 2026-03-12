"""Redis Stack vector indexing for semantic search."""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import numpy as np
import redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from rich.console import Console

from shard.config import ShardConfig, check_redis
from shard.pipeline import FormattedNote, IndexedNote, IndexingError, SourceType
from shard.vault import list_shards, parse_frontmatter, read_note

logger = logging.getLogger(__name__)
_console = Console(stderr=True)

_EMBEDDING_LOAD_TIMEOUT = 120  # seconds; covers first-time model download
_INDEX_NAME = "shard-chunks"
_KEY_PREFIX = "shard:chunk:"
_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2; must match config.embedding_model


# ── Text chunking ─────────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split *text* into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []

    overlap = min(overlap, chunk_size - 1)
    step = chunk_size - overlap

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += step

    return chunks


# ── Embedding model ───────────────────────────────────────────────────────────


def _load_embedding_model(model_name: str):
    """Load the sentence-transformers model with a timeout."""
    from sentence_transformers import SentenceTransformer

    def _load():
        return SentenceTransformer(model_name)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_load)
        try:
            return future.result(timeout=_EMBEDDING_LOAD_TIMEOUT)
        except FuturesTimeoutError:
            raise IndexingError(
                f"Timed out loading embedding model '{model_name}' "
                f"(>{_EMBEDDING_LOAD_TIMEOUT}s). The model may be downloading. "
                "Check your network connection or try again."
            )


def _encode_vector(embedding: np.ndarray) -> bytes:
    """Serialize a numpy float32 embedding to bytes for Redis."""
    return embedding.astype(np.float32).tobytes()


# ── Redis client ──────────────────────────────────────────────────────────────


def get_redis_client(config: ShardConfig) -> redis.Redis:
    """Create a Redis client with connection pooling."""
    pool = redis.ConnectionPool(
        host=config.redis_host,
        port=config.redis_port,
        password=config.redis_password or None,
        decode_responses=False,
    )
    return redis.Redis(connection_pool=pool)


def _ensure_index(client: redis.Redis) -> None:
    """Create the shard-chunks RediSearch index if it doesn't exist."""
    try:
        client.ft(_INDEX_NAME).info()
        # Index exists — verify basic compatibility
        return
    except redis.ResponseError:
        pass  # Index doesn't exist, create it

    schema = (
        TextField("content"),
        TagField("source_file"),
        TextField("source_path"),
        TextField("title"),
        TagField("tags", separator=" "),
        TagField("source_type"),
        NumericField("date_added", sortable=True),
        NumericField("chunk_index"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": _EMBEDDING_DIM,
                "DISTANCE_METRIC": "COSINE",
                "M": 16,
                "EF_CONSTRUCTION": 200,
            },
        ),
    )

    definition = IndexDefinition(
        prefix=[_KEY_PREFIX],
        index_type=IndexType.HASH,
    )

    try:
        client.ft(_INDEX_NAME).create_index(
            fields=schema,
            definition=definition,
        )
    except redis.ResponseError as exc:
        if "Index already exists" not in str(exc):
            raise IndexingError(f"Failed to create Redis index: {exc}") from exc


# ── Note indexing ─────────────────────────────────────────────────────────────


def index_note(
    note: FormattedNote,
    path: Path,
    config: ShardConfig,
    client: redis.Redis | None = None,
    model=None,
) -> IndexedNote:
    """Chunk *note* and store all chunks in Redis with vector embeddings."""
    try:
        if client is None:
            if not check_redis(config):
                raise IndexingError("Redis is not available")
            client = get_redis_client(config)
            _ensure_index(client)

        if model is None:
            model = _load_embedding_model(config.embedding_model)

        chunks = chunk_text(note.body)
        if not chunks:
            chunks = [note.summary or note.title]

        embeddings = model.encode(chunks)

        pipe = client.pipeline(transaction=False)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            key = f"{_KEY_PREFIX}{uuid.uuid4()}"
            mapping = {
                "content": chunk,
                "source_file": path.name,
                "source_path": str(path),
                "title": note.title,
                "tags": " ".join(note.tags),
                "source_type": note.source_type.name.lower(),
                "date_added": int(time.time()),
                "chunk_index": i,
                "embedding": _encode_vector(embedding),
            }
            pipe.hset(key, mapping=mapping)
        pipe.execute()

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
        raise IndexingError(f"Failed to index note '{path}': {exc}") from exc


# ── Vault reindex ─────────────────────────────────────────────────────────────


def _drop_all_chunk_keys(client: redis.Redis) -> int:
    """Delete all shard:chunk:* keys. Returns count of deleted keys."""
    count = 0
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, match=f"{_KEY_PREFIX}*", count=500)
        if keys:
            client.delete(*keys)
            count += len(keys)
        if cursor == 0:
            break
    return count


def reindex_vault(config: ShardConfig) -> int:
    """Reindex every Shard note in the vault into Redis."""
    from shard.ui.status import StatusFeed

    if not check_redis(config):
        raise IndexingError("Redis is not available. See above for setup instructions.")

    note_paths = list_shards(config)
    total_chunks = 0

    if not note_paths:
        _console.print("[yellow]No shard notes found — nothing to index.[/yellow]")
        return 0

    client = get_redis_client(config)

    with StatusFeed() as status:
        status.update("Clearing existing index...")
        _drop_all_chunk_keys(client)

        # Drop and recreate index
        try:
            client.ft(_INDEX_NAME).dropindex(delete_documents=False)
        except redis.ResponseError:
            pass
        _ensure_index(client)

        status.update("Loading embedding model...")
        model = _load_embedding_model(config.embedding_model)

        for i, note_path in enumerate(note_paths, 1):
            status.update(f"Embedding note {i}/{len(note_paths)}: {note_path.name}...")

            content = read_note(note_path)
            metadata, body = parse_frontmatter(content)

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

            indexed = index_note(note, note_path, config, client=client, model=model)
            total_chunks += indexed.num_chunks

    return total_chunks
