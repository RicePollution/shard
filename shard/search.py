"""Semantic Q&A search over indexed notes using Redis Stack and LLM completion."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from shard import models
from shard.config import ShardConfig, check_redis
from shard.pipeline import IndexingError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AskResult:
    """Container for a Q&A response with source attributions."""

    answer: str
    sources: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# QA system prompt
# ---------------------------------------------------------------------------

_QA_SYSTEM = (
    "You are a helpful research assistant. Answer the user's question based "
    "ONLY on the provided context excerpts from their personal notes. Follow "
    "these rules strictly:\n"
    "- Cite which source(s) your information comes from by mentioning their "
    "title or filename.\n"
    "- If the context does not contain enough information to answer the "
    'question, say "I don\'t have enough information in your notes to answer '
    'that."\n'
    "- Do not fabricate information that is not present in the context."
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ask(
    question: str,
    config: ShardConfig,
    top_k: int = 5,
    on_status: Callable[[str], None] | None = None,
) -> AskResult:
    """Answer *question* by searching indexed notes and prompting the LLM."""
    from redis.commands.search.query import Query

    from shard.pipeline.indexer import (
        _INDEX_NAME,
        _encode_vector,
        _load_embedding_model,
        get_redis_client,
    )

    # -- 1. Check Redis availability ----------------------------------------

    if not check_redis(config):
        raise IndexingError("Redis is not available. See above for setup instructions.")

    if on_status:
        on_status("Searching vault...")

    client = get_redis_client(config)

    # -- 2. Check if index exists and has documents -------------------------

    try:
        info = client.ft(_INDEX_NAME).info()
        num_docs = int(info.get("num_docs", info.get(b"num_docs", 0)))
    except Exception:
        return AskResult(
            answer="No notes indexed yet. Run 'shard add' to import content.",
            sources=[],
        )

    if num_docs == 0:
        return AskResult(
            answer="No notes indexed yet. Run 'shard add' to import content.",
            sources=[],
        )

    # -- 3. Embed the question ----------------------------------------------

    model = _load_embedding_model(config.embedding_model)
    query_embedding = model.encode(question)
    query_bytes = _encode_vector(query_embedding)

    # -- 4. KNN vector search -----------------------------------------------

    k = min(top_k, num_docs)

    q = (
        Query(f"*=>[KNN {k} @embedding $vec AS vector_score]")
        .sort_by("vector_score")
        .return_fields("content", "title", "source_path", "source_file", "vector_score")
        .paging(0, k)
        .dialect(2)
    )

    try:
        results = client.ft(_INDEX_NAME).search(q, query_params={"vec": query_bytes})
    except Exception as exc:
        logger.error("Redis search failed: %s", exc)
        return AskResult(
            answer="Search failed. Try running 'shard index' to rebuild the index.",
            sources=[],
        )

    if not results.docs:
        return AskResult(
            answer="No relevant notes found for your question.",
            sources=[],
        )

    if on_status:
        on_status(f"Found {len(results.docs)} relevant chunks — generating answer...")

    # -- 5. Build context string --------------------------------------------

    context_parts: list[str] = []
    raw_sources: list[tuple[str, str, float]] = []

    for idx, doc in enumerate(results.docs, start=1):
        content = doc.content if hasattr(doc, "content") else ""
        title = doc.title if hasattr(doc, "title") else "Untitled"
        source_path = doc.source_path if hasattr(doc, "source_path") else "unknown"
        score_raw = float(doc.vector_score) if hasattr(doc, "vector_score") else 1.0

        # Cosine distance is in [0, 2]; convert to relevance in [0, 1]
        relevance = round(1.0 - (score_raw / 2.0), 4)

        context_parts.append(f"[Source {idx}: {title} ({source_path})]\n{content}")
        raw_sources.append((title, source_path, relevance))

    context_block = "\n\n---\n\n".join(context_parts)

    # -- 6. Build prompt and call the LLM -----------------------------------

    prompt = (
        f"Context from notes:\n\n{context_block}\n\n---\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above. Cite which "
        "source(s) your answer comes from."
    )

    answer = models.complete(prompt=prompt, system=_QA_SYSTEM)

    # -- 7. Extract unique sources ------------------------------------------

    seen_keys: set[str] = set()
    sources: list[dict] = []

    for title, path, relevance in raw_sources:
        key = f"{title}::{path}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        sources.append({
            "title": title,
            "path": path,
            "relevance_score": relevance,
        })

    # -- 8. Return ----------------------------------------------------------

    return AskResult(answer=answer, sources=sources)
