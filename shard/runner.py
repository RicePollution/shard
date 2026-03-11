"""Pipeline orchestration — sequences extract → format → index."""

from __future__ import annotations

from rich.console import Console

from shard.config import ShardConfig, get_config
from shard.pipeline import IndexedNote
from shard.pipeline.extractor import extract
from shard.pipeline.formatter import format_notes
from shard.pipeline.indexer import _get_collection, index_note
from shard.vault import save_note

_console = Console(stderr=True)


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

    with _console.status("[bold cyan]Extracting content…[/bold cyan]", spinner="dots"):
        extracted = extract(input_str)
    source_label = f"{extracted.source_type.name.lower()} — {extracted.title}"
    _console.print(f"[green]Extracted:[/green] {source_label}")

    with _console.status("[bold cyan]Formatting notes…[/bold cyan]", spinner="dots"):
        formatted_notes = format_notes(extracted, single=single)
    if single:
        _console.print(
            f"[green]Formatted:[/green] {formatted_notes[0].title} "
            f"({len(formatted_notes[0].tags)} tags)"
        )
    else:
        _console.print(
            f"[green]Formatted:[/green] Split into {len(formatted_notes)} atomic notes"
        )

    # Build the ChromaDB collection once (loads embedding model) and reuse it.
    with _console.status("[bold cyan]Loading index…[/bold cyan]", spinner="dots"):
        collection = _get_collection(config)

    indexed_notes: list[IndexedNote] = []

    for i, formatted in enumerate(formatted_notes, 1):
        label = f"{i}/{len(formatted_notes)}"
        with _console.status(
            f"[bold cyan]Saving note {label}…[/bold cyan]", spinner="dots"
        ):
            path = save_note(formatted, config)
        _console.print(
            f"[green]Saved:[/green] {path.relative_to(config.vault_path)} ({label})"
        )

        with _console.status(
            f"[bold cyan]Indexing {label}…[/bold cyan]", spinner="dots"
        ):
            indexed = index_note(formatted, path, config, collection=collection)
        indexed_notes.append(indexed)

    total_chunks = sum(n.num_chunks for n in indexed_notes)
    _console.print(f"[green]Indexed:[/green] {total_chunks} chunks total")

    return indexed_notes
