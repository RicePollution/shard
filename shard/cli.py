"""Shard CLI entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from shard.config import (
    CONFIG_PATH,
    first_run_setup,
    get_config,
    save_config,
)
from shard.pipeline import ShardError

# stderr console for status messages and errors; stdout console for primary output.
_err = Console(stderr=True)
_out = Console()


# ── Group ─────────────────────────────────────────────────────────────────────


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Shard — AI-powered note ingestion for Obsidian.

    Ingest PDFs, URLs, YouTube videos, and plain text into your Obsidian vault
    as structured, AI-generated notes with full semantic search support.
    """
    ctx.ensure_object(dict)

    # First-run detection: if the config file has never been written, walk the
    # user through the interactive setup wizard before any command runs.
    if not CONFIG_PATH.exists():
        try:
            config = first_run_setup()
        except click.Abort:
            _err.print("\n[yellow]Setup cancelled.[/yellow]")
            sys.exit(1)
        ctx.obj["config"] = config
    else:
        # Defer loading; commands that need it will call get_config() directly.
        ctx.obj["config"] = None


# ── add ───────────────────────────────────────────────────────────────────────


@cli.command("add")
@click.argument("input", metavar="INPUT")
def add(input: str) -> None:
    """Add a note from a file path, URL, or raw text.

    INPUT can be a local file path (PDF or text), a web URL, a YouTube URL,
    or any freeform text string to import directly.
    """
    from shard.runner import run_add_pipeline

    try:
        config = get_config()
        indexed = run_add_pipeline(input, config=config)
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    # Primary result summary goes to stdout so it can be piped/captured.
    _out.print()
    _out.print("[bold green]Shard added successfully.[/bold green]")
    _out.print(f"  [dim]Title:[/dim]  {indexed.title}")
    _out.print(f"  [dim]Tags:[/dim]   {', '.join(indexed.tags) if indexed.tags else '(none)'}")
    _out.print(f"  [dim]Chunks:[/dim] {indexed.num_chunks}")
    _out.print(f"  [dim]Path:[/dim]   {indexed.path}")


# ── ask ───────────────────────────────────────────────────────────────────────


@cli.command("ask")
@click.argument("question", metavar="QUESTION")
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    metavar="N",
    help="Number of note chunks to retrieve as context.",
)
def ask(question: str, top_k: int) -> None:
    """Ask a question and get an answer sourced from your notes.

    QUESTION is a natural-language query.  Shard retrieves the most relevant
    chunks from indexed notes and uses your configured LLM to generate a
    grounded answer.
    """
    from shard.search import ask as search_ask

    try:
        config = get_config()
        with _err.status("[bold cyan]Searching notes…[/bold cyan]", spinner="dots"):
            result = search_ask(question, config=config, top_k=top_k)
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    # Answer block
    _out.print()
    _out.print(result.answer)

    if not result.sources:
        return

    # Sources table
    _out.print()
    table = Table(
        title="Sources",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
        expand=False,
    )
    table.add_column("Title", style="white", no_wrap=False)
    table.add_column("Path", style="dim", no_wrap=False)
    table.add_column("Relevance", justify="right", style="green", no_wrap=True)

    for source in result.sources:
        relevance_pct = f"{source['relevance_score'] * 100:.1f}%"
        table.add_row(
            source.get("title", "Untitled"),
            source.get("path", ""),
            relevance_pct,
        )

    _out.print(table)


# ── index ─────────────────────────────────────────────────────────────────────


@cli.command("index")
def index() -> None:
    """Reindex all shard notes in the vault into ChromaDB.

    Iterates every note under <vault>/Imported/Shards/, chunks its content,
    and upserts the chunks into the local ChromaDB collection so they are
    available for semantic search.
    """
    from shard.pipeline.indexer import reindex_vault

    try:
        config = get_config()
        total = reindex_vault(config)
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    _out.print(f"\n[bold green]Reindex complete.[/bold green] Total chunks: [bold]{total}[/bold]")


# ── list ──────────────────────────────────────────────────────────────────────


@cli.command("list")
def list_() -> None:
    """List all imported shard notes in the vault.

    Reads the YAML frontmatter of each note and displays a summary table
    with title, tags, creation date, and vault-relative path.
    """
    from shard.vault import list_shards, parse_frontmatter, read_note

    try:
        config = get_config()
        paths = list_shards(config)
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    if not paths:
        _err.print(
            "[yellow]No shard notes found.[/yellow] "
            "Run [bold]shard add[/bold] to import content."
        )
        return

    table = Table(
        title=f"Shard Notes ({len(paths)})",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
        expand=True,
    )
    table.add_column("Title", style="white", ratio=3)
    table.add_column("Tags", style="cyan", ratio=2)
    table.add_column("Date", style="dim", no_wrap=True)
    table.add_column("Path", style="dim", ratio=3)

    for path in paths:
        try:
            content = read_note(path)
            metadata, _ = parse_frontmatter(content)
        except ShardError:
            metadata = {}

        title = metadata.get("title", path.stem)
        raw_tags = metadata.get("tags", [])
        tags_str = ", ".join(raw_tags) if isinstance(raw_tags, list) else str(raw_tags)
        date = metadata.get("date", "")

        try:
            rel_path = str(path.relative_to(config.vault_path))
        except ValueError:
            rel_path = str(path)

        table.add_row(title, tags_str, date, rel_path)

    _out.print(table)


# ── open ──────────────────────────────────────────────────────────────────────


@cli.command("open")
@click.argument("query", metavar="QUERY")
def open_(query: str) -> None:
    """Open the best-matching shard note in Obsidian.

    QUERY is a fuzzy search string matched against note titles.  The note with
    the highest partial-ratio score is opened via the obsidian:// URI scheme.
    """
    from thefuzz import process as fuzz_process

    from shard.vault import list_shards, open_in_obsidian, parse_frontmatter, read_note

    try:
        config = get_config()
        paths = list_shards(config)
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    if not paths:
        _err.print(
            "[yellow]No shard notes found.[/yellow] "
            "Run [bold]shard add[/bold] to import content."
        )
        sys.exit(1)

    # Build a mapping of title -> Path for fuzzy matching.
    title_to_path: dict[str, Path] = {}
    for path in paths:
        try:
            content = read_note(path)
            metadata, _ = parse_frontmatter(content)
            title = metadata.get("title", path.stem)
        except ShardError:
            title = path.stem
        # If multiple notes share a title, prefer the later one (already sorted).
        title_to_path[title] = path

    if not title_to_path:
        _err.print("[red]Could not read any note titles.[/red]")
        sys.exit(1)

    match = fuzz_process.extractOne(query, title_to_path.keys())

    if match is None:
        _err.print(f"[bold red]No match found[/bold red] for query: [italic]{query}[/italic]")
        sys.exit(1)

    matched_title, score, *_ = match

    if score < 40:
        _err.print(
            f"[bold red]No confident match found[/bold red] for [italic]{query}[/italic]. "
            f"Best candidate was [italic]{matched_title}[/italic] (score {score}/100)."
        )
        sys.exit(1)

    matched_path = title_to_path[matched_title]
    _err.print(f"[dim]Matched:[/dim] [bold]{matched_title}[/bold] [dim](score {score}/100)[/dim]")

    try:
        open_in_obsidian(matched_path, config)
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    _err.print("[green]Opened[/green] in Obsidian.")


# ── config ────────────────────────────────────────────────────────────────────

_SETTABLE_FIELDS = {
    "vault_path",
    "model",
    "chroma_path",
    "embedding_model",
}

_FIELD_HELP = {
    "vault_path": "Absolute path to your Obsidian vault directory.",
    "model": "LiteLLM model string used for note generation (e.g. ollama_chat/qwen2.5:3b).",
    "chroma_path": "Directory where ChromaDB persists its vector index.",
    "embedding_model": "Sentence-transformers model name for vector embeddings.",
}


@cli.command("config")
@click.option(
    "--set",
    "set_value",
    metavar="KEY=VALUE",
    default=None,
    help=(
        "Set a single config field.  Accepted keys: "
        + ", ".join(sorted(_SETTABLE_FIELDS))
        + "."
    ),
)
@click.option(
    "--setup",
    "rerun_setup",
    is_flag=True,
    default=False,
    help="Rerun the interactive first-run setup wizard.",
)
def config(set_value: str | None, rerun_setup: bool) -> None:
    """Show or update Shard configuration.

    With no flags, displays all current configuration values.

    Use --set KEY=VALUE to update a single field without re-running the wizard.
    Use --setup to rerun the interactive first-run setup wizard from scratch.
    """
    if rerun_setup:
        try:
            first_run_setup()
        except click.Abort:
            _err.print("\n[yellow]Setup cancelled.[/yellow]")
            sys.exit(1)
        return

    if set_value is not None:
        _handle_config_set(set_value)
        return

    # Display current configuration.
    _display_config()


def _handle_config_set(raw: str) -> None:
    """Parse KEY=VALUE and update the persisted config."""
    if "=" not in raw:
        _err.print(
            "[bold red]Error:[/bold red] --set requires KEY=VALUE format "
            f"(e.g. --set model=ollama_chat/llama3). Got: [italic]{raw}[/italic]"
        )
        sys.exit(1)

    key, _, value = raw.partition("=")
    key = key.strip()
    value = value.strip()

    if key not in _SETTABLE_FIELDS:
        _err.print(
            f"[bold red]Error:[/bold red] Unknown config key [bold]{key!r}[/bold]. "
            "Settable fields: " + ", ".join(sorted(_SETTABLE_FIELDS)) + "."
        )
        sys.exit(1)

    try:
        cfg = get_config()
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    # Coerce Path fields.
    if key in ("vault_path", "chroma_path"):
        coerced: object = Path(value).expanduser().resolve()
    else:
        coerced = value

    setattr(cfg, key, coerced)

    try:
        save_config(cfg)
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    _err.print(f"[green]Config updated:[/green] [bold]{key}[/bold] = [italic]{coerced}[/italic]")
    _err.print(f"[dim]Saved to {CONFIG_PATH}[/dim]")


def _display_config() -> None:
    """Print the current config as a Rich table."""
    try:
        cfg = get_config()
    except ShardError as exc:
        _err.print(f"[bold red]Error loading config:[/bold red] {exc}")
        _err.print(f"[dim]Expected config at {CONFIG_PATH}[/dim]")
        sys.exit(1)

    table = Table(
        title=f"Shard Configuration  [dim]{CONFIG_PATH}[/dim]",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
        expand=False,
    )
    table.add_column("Key", style="bold white", no_wrap=True)
    table.add_column("Value", style="green", no_wrap=False)
    table.add_column("Description", style="dim", no_wrap=False)

    rows = [
        ("vault_path",      str(cfg.vault_path)),
        ("model",           cfg.model or "[dim](not set)[/dim]"),
        ("chroma_path",     str(cfg.chroma_path)),
        ("embedding_model", cfg.embedding_model),
    ]

    for key, value in rows:
        desc = _FIELD_HELP.get(key, "")
        table.add_row(key, value, desc)

    # Custom models section — only show when present.
    if cfg.custom_models:
        table.add_section()
        for entry in cfg.custom_models:
            name = entry.get("name", "")
            provider = entry.get("provider", "")
            has_key = bool(cfg.api_keys.get(provider))
            availability = "[green]key set[/green]" if has_key else "[yellow]no key[/yellow]"
            table.add_row(
                "custom_model",
                f"{name} [dim]({provider})[/dim] {availability}",
                "User-registered custom model.",
            )

    _out.print(table)

    # API keys: show provider names but never the key values.
    if cfg.api_keys:
        _out.print()
        providers = ", ".join(sorted(cfg.api_keys.keys()))
        _out.print(f"[bold]Configured API key providers:[/bold] {providers}")
