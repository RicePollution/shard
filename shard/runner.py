"""Pipeline orchestration — sequences extract → format → index."""

from __future__ import annotations

from shard.config import ShardConfig, get_config
from shard.pipeline import IndexedNote
from shard.pipeline.extractor import extract
from shard.pipeline.formatter import format_notes
from shard.pipeline.indexer import _get_collection, index_note
from shard.ui.status import StatusFeed
from shard.vault import save_note


def run_add_pipeline(
    input_str: str,
    config: ShardConfig | None = None,
    single: bool = False,
) -> list[IndexedNote]:
    """Run the full add pipeline: extract → format → save → index.

    Parameters
    ----------
    input_str:
        The raw input (file path, URL, or text) to process.
    config:
        Optional config override; defaults to :func:`~shard.config.get_config`.
    single:
        When True, bypasses atomic splitting and generates a single note.

    Returns
    -------
    list[IndexedNote]
        The indexed notes with path, title, tags, summary, and chunk count.

    Raises
    ------
    ShardError
        Any pipeline stage error propagates as a :class:`ShardError` subclass.
    """
    if config is None:
        config = get_config()

    with StatusFeed() as status:
        status.update("Detecting source type...")
        extracted = extract(input_str)
        source_label = extracted.source_type.name.lower()

        status.update(f"Extracting content from {source_label}...")

        formatted_notes = format_notes(extracted, single=single, on_status=status.update)

        status.update("Loading search index...")
        collection = _get_collection(config)

        indexed_notes: list[IndexedNote] = []
        total = len(formatted_notes)

        for i, formatted in enumerate(formatted_notes, 1):
            status.update(f"Saving and indexing note {i}/{total}: {formatted.title}...")
            path = save_note(formatted, config)
            indexed = index_note(formatted, path, config, collection=collection)
            indexed_notes.append(indexed)

    return indexed_notes
