"""Pipeline orchestration — sequences extract → format → index."""

from __future__ import annotations

from rich.console import Console

from shard.config import ShardConfig, get_config
from shard.pipeline import IndexedNote
from shard.pipeline.extractor import extract
from shard.pipeline.formatter import format_note
from shard.pipeline.indexer import index_note
from shard.vault import save_note

_console = Console(stderr=True)


def run_add_pipeline(input_str: str, config: ShardConfig | None = None) -> IndexedNote:
    """Run the full add pipeline: extract → format → save → index.

    Parameters
    ----------
    input_str:
        The raw input (file path, URL, or text) to process.
    config:
        Optional config override; defaults to :func:`~shard.config.get_config`.

    Returns
    -------
    IndexedNote
        The final indexed note with path, title, tags, summary, and chunk count.

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

    with _console.status("[bold cyan]Formatting note…[/bold cyan]", spinner="dots"):
        formatted = format_note(extracted)
    _console.print(f"[green]Formatted:[/green] {formatted.title} ({len(formatted.tags)} tags)")

    with _console.status("[bold cyan]Saving to vault…[/bold cyan]", spinner="dots"):
        path = save_note(formatted, config)
    _console.print(f"[green]Saved:[/green] {path.relative_to(config.vault_path)}")

    with _console.status("[bold cyan]Indexing…[/bold cyan]", spinner="dots"):
        indexed = index_note(formatted, path, config)
    _console.print(f"[green]Indexed:[/green] {indexed.num_chunks} chunks")

    return indexed
