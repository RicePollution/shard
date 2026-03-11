"""Semantic Q&A search over indexed notes using ChromaDB and LLM completion."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import chromadb

from shard import models
from shard.config import ShardConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AskResult:
    """Container for a Q&A response with source attributions.

    Attributes:
        answer: The generated answer text.
        sources: A list of source dicts, each containing keys such as
            ``title``, ``path``, and ``relevance_score``.
    """

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
    """Answer *question* by searching indexed notes and prompting the LLM.

    Parameters
    ----------
    question:
        The natural-language question to answer.
    config:
        A :class:`~shard.config.ShardConfig` providing ``chroma_path`` and
        ``embedding_model``.
    top_k:
        Maximum number of note chunks to retrieve from ChromaDB.

    Returns
    -------
    AskResult
        The AI-generated answer together with a list of source attributions.
    """

    # -- 1. Connect to ChromaDB and get the collection --------------------

    from shard.pipeline.indexer import _load_embedding_fn

    if on_status:
        on_status("Searching vault...")

    embedding_fn = _load_embedding_fn(config.embedding_model)

    client = chromadb.PersistentClient(path=str(config.chroma_path))

    try:
        collection = client.get_collection(
            name="shard_notes",
            embedding_function=embedding_fn,
        )
    except Exception:
        # Collection does not exist yet.
        return AskResult(
            answer=(
                "No notes indexed yet. Run 'shard add' to import content."
            ),
            sources=[],
        )

    # -- 2. Handle empty collection ---------------------------------------

    if collection.count() == 0:
        return AskResult(
            answer="No notes indexed yet. Run 'shard add' to import content.",
            sources=[],
        )

    # -- 3. Query ChromaDB ------------------------------------------------

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
    )

    documents: list[str] = results.get("documents", [[]])[0]
    metadatas: list[dict] = results.get("metadatas", [[]])[0]
    distances: list[float] = results.get("distances", [[]])[0]

    if not documents:
        return AskResult(
            answer="No relevant notes found for your question.",
            sources=[],
        )

    if on_status:
        on_status(f"Found {len(documents)} relevant chunks — generating answer...")

    # -- 4. Build context string ------------------------------------------

    context_parts: list[str] = []
    for idx, (text, meta) in enumerate(zip(documents, metadatas), start=1):
        title = meta.get("title", "Untitled")
        source = meta.get("source", "unknown")
        context_parts.append(
            f"[Source {idx}: {title} ({source})]\n{text}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    # -- 5. Build prompt and call the LLM ---------------------------------

    prompt = (
        f"Context from notes:\n\n{context_block}\n\n---\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above. Cite which "
        "source(s) your answer comes from."
    )

    answer = models.complete(prompt=prompt, system=_QA_SYSTEM)

    # -- 6. Extract unique sources ----------------------------------------

    seen_keys: set[str] = set()
    sources: list[dict] = []

    for meta, distance in zip(metadatas, distances):
        title = meta.get("title", "Untitled")
        path = meta.get("source", "unknown")

        key = f"{title}::{path}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # ChromaDB returns L2 distances; convert to a 0-1 relevance score.
        # Smaller distance == higher relevance.  We use 1/(1+d) as a simple
        # monotonically-decreasing mapping.
        relevance_score = round(1.0 / (1.0 + distance), 4)

        sources.append({
            "title": title,
            "path": path,
            "relevance_score": relevance_score,
        })

    # -- 7. Return --------------------------------------------------------

    return AskResult(answer=answer, sources=sources)
