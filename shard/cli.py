"""Shard CLI entry point."""

from __future__ import annotations

import os
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
@click.option(
    "--single",
    is_flag=True,
    default=False,
    help="Generate one note instead of splitting into atomic notes.",
)
def add(input: str, single: bool) -> None:
    """Add a note from a file path, URL, or raw text.

    INPUT can be a local file path (PDF or text), a web URL, a YouTube URL,
    or any freeform text string to import directly.

    By default, content is split into multiple atomic notes (one idea per note)
    that are interlinked with [[wikilinks]]. Use --single to generate a single
    note instead.
    """
    from shard.runner import run_add_pipeline

    try:
        config = get_config()
        indexed_notes = run_add_pipeline(input, config=config, single=single)
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    _out.print()
    if len(indexed_notes) == 1:
        # Single note output (--single or fallback)
        indexed = indexed_notes[0]
        _out.print("[bold green]Shard added successfully.[/bold green]")
        _out.print(f"  [dim]Title:[/dim]  {indexed.title}")
        _out.print(f"  [dim]Tags:[/dim]   {', '.join(indexed.tags) if indexed.tags else '(none)'}")
        _out.print(f"  [dim]Chunks:[/dim] {indexed.num_chunks}")
        _out.print(f"  [dim]Path:[/dim]   {indexed.path}")
    else:
        # Atomic notes output
        _out.print(f"[bold green]✓ Split into {len(indexed_notes)} atomic notes:[/bold green]")
        for i, indexed in enumerate(indexed_notes):
            label = "(parent index)" if i == 0 else ""
            _out.print(f"  📄 {indexed.path.name:<30s} {label}")


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


# ── learn ─────────────────────────────────────────────────────────────────────

STYLE_PROFILE_PATH = Path.home() / ".shard" / "style.json"


@cli.command("learn")
@click.option("--force", is_flag=True, default=False, help="Re-analyze even if a style profile exists.")
@click.option("--show", "show_profile", is_flag=True, default=False, help="Print current style fingerprint.")
@click.option("--template", "show_template", is_flag=True, default=False, help="Print the blank note template.")
@click.option(
    "--depth",
    type=click.Choice(["quick", "normal", "deep"], case_sensitive=False),
    default="normal",
    show_default=True,
    help="Analysis depth: quick (5 notes, 1 call), normal (20 notes), deep (all notes).",
)
def learn(force: bool, show_profile: bool, show_template: bool, depth: str) -> None:
    """Learn your note-writing style from existing vault notes.

    Analyzes your vault and saves a style profile so future notes from
    shard add match your writing style exactly.
    """
    from shard.pipeline.learner import Learner, load_style_profile, save_style_profile
    from shard.vault import read_note, walk_vault

    # --show: display current fingerprints
    if show_profile:
        profile = load_style_profile(STYLE_PROFILE_PATH)
        if profile is None:
            _err.print("[yellow]No style profile found.[/yellow] Run [bold]shard learn[/bold] first.")
            sys.exit(1)
        _print_style_summary(profile)
        return

    # --template: display blank note template
    if show_template:
        profile = load_style_profile(STYLE_PROFILE_PATH)
        if profile is None:
            _err.print("[yellow]No style profile found.[/yellow] Run [bold]shard learn[/bold] first.")
            sys.exit(1)
        _out.print(profile.template)
        return

    # Check for existing profile
    if not force and STYLE_PROFILE_PATH.exists():
        _err.print(
            "[yellow]Style profile already exists.[/yellow] "
            "Use [bold]--force[/bold] to re-analyze."
        )
        profile = load_style_profile(STYLE_PROFILE_PATH)
        if profile:
            _print_style_summary(profile)
        return

    import random

    # Import here to match existing lazy-import pattern
    from shard.pipeline.learner import QUICK_SAMPLE_SIZE, MAX_SAMPLE_SIZE

    try:
        config = get_config()
        with _err.status("[bold cyan]Sampling vault notes…[/bold cyan]", spinner="dots"):
            paths = walk_vault(config)
            notes = []
            for p in paths:
                try:
                    notes.append(read_note(p))
                except ShardError:
                    continue

        if len(notes) < 5:
            _err.print(
                f"[yellow]Not enough notes to learn style (found {len(notes)}, need 5+).[/yellow]\n"
                "Add some notes first, then re-run [bold]shard learn[/bold]."
            )
            sys.exit(1)

        # Determine sample size based on depth
        if depth == "quick":
            sample_size = min(len(notes), QUICK_SAMPLE_SIZE)
            api_calls = 1
        elif depth == "deep":
            sample_size = len(notes)
            api_calls = len(notes) + 1
        else:
            sample_size = min(len(notes), MAX_SAMPLE_SIZE)
            api_calls = sample_size + 1

        _err.print(
            f"[green]●[/green] Mode: {depth} ({sample_size} notes, ~{api_calls} API calls)"
        )

        # Deep mode confirmation for large vaults
        if depth == "deep" and len(notes) >= 50:
            _err.print(
                f"\n[yellow]⚠️  Deep analysis on {len(notes)} notes will make "
                f"{len(notes)} API calls.[/yellow]\n"
                "This may take several minutes and use significant credits."
            )
            if not click.confirm("Continue?", default=False):
                _err.print("[yellow]Aborted.[/yellow]")
                return

        learner = Learner()

        if depth == "quick":
            # Quick: single pass, no progress needed for individual notes
            with _err.status("[bold cyan]● Analyzing style…[/bold cyan]", spinner="dots"):
                profile = learner.analyze(notes, depth="quick")
            _err.print("[green]●[/green] Quick analysis...              [green]✓[/green]")
        else:
            # Normal / Deep: use all notes for deep, sampled for normal
            if depth == "deep":
                sampled = list(notes)
            elif len(notes) > MAX_SAMPLE_SIZE:
                sampled = random.sample(notes, MAX_SAMPLE_SIZE)
            else:
                sampled = list(notes)

            _err.print(
                f"[green]●[/green] Sampling vault notes...        "
                f"[green]✓[/green] ({len(sampled)} notes sampled)"
            )

            # Pass 1 with progress
            with _err.status(
                f"[bold cyan]● Pass 1: Analyzing structure… 0/{len(sampled)}[/bold cyan]",
                spinner="dots",
            ):
                pass1_results = learner._pass1_extract(sampled)

            _err.print(
                f"[green]●[/green] Pass 1: Analyzing structure... "
                f"[green]✓[/green] {len(pass1_results)}/{len(pass1_results)}"
            )

            # Pass 2
            with _err.status("[bold cyan]● Pass 2: Synthesizing style…[/bold cyan]", spinner="dots"):
                profile = learner._pass2_synthesize(pass1_results, len(pass1_results))

            _err.print("[green]●[/green] Pass 2: Synthesizing style...  [green]✓[/green]")

        save_style_profile(profile, STYLE_PROFILE_PATH)
        _err.print("[green]●[/green] Style profile saved            [green]✓[/green]")

    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    _out.print()
    _print_style_summary(profile)
    _out.print(f"\n  Average note length: ~{profile.avg_word_count} words")
    _out.print("  Future notes will match your style exactly.")
    _out.print("  Re-run [bold]shard learn[/bold] anytime to update.")


def _print_style_summary(profile: "StyleProfile") -> None:  # noqa: F821
    """Print a rich panel showing the style fingerprint."""
    from rich.panel import Panel

    lines = []
    for i, fp in enumerate(profile.fingerprints, 1):
        lines.append(f"  {i}. {fp}")
    content = "\n".join(lines)

    panel = Panel(
        content,
        title="[bold]📝 Your note fingerprint[/bold]",
        expand=False,
    )
    _out.print(panel)


# ── sync ──────────────────────────────────────────────────────────────────────


@cli.command("sync")
@click.option("--dry-run", is_flag=True, default=False, help="Preview links without changing files.")
@click.option("--vault", "vault_override", default=None, metavar="PATH", help="Override vault path.")
@click.option("--verbose", is_flag=True, default=False, help="Show each link as it's added.")
def sync(dry_run: bool, vault_override: str | None, verbose: bool) -> None:
    """Sync backlinks across your vault.

    Scans all notes and adds [[wikilinks]] between related notes to build
    a connected knowledge graph. Always creates a backup before making changes.
    """
    import shutil
    from datetime import datetime, timezone

    from shard.pipeline.linker import Linker, apply_links
    from shard.vault import parse_frontmatter, read_note, walk_vault

    try:
        import copy

        config = copy.copy(get_config())
        if vault_override:
            config.vault_path = Path(vault_override).expanduser().resolve()

        # Backup
        if not dry_run:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            backup_dir = Path.home() / ".shard" / "backups" / timestamp
            with _err.status("[bold cyan]Backing up vault…[/bold cyan]", spinner="dots"):
                backup_dir.mkdir(parents=True, exist_ok=True)
                for md_file in walk_vault(config):
                    rel = md_file.relative_to(config.vault_path)
                    dest = backup_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(md_file, dest)
            _err.print("[green]●[/green] Backing up vault...            [green]✓[/green]")

        # Build title index
        with _err.status("[bold cyan]Building title index…[/bold cyan]", spinner="dots"):
            all_paths = walk_vault(config)
            title_map: dict[str, Path] = {}
            for p in all_paths:
                try:
                    content = read_note(p)
                    metadata, _ = parse_frontmatter(content)
                    title = metadata.get("title", p.stem)
                except ShardError:
                    title = p.stem
                title_map[title] = p

        all_titles = list(title_map.keys())
        _err.print(
            f"[green]●[/green] Building title index...        "
            f"[green]✓[/green] ({len(all_titles)} notes)"
        )

        # Process notes in batches
        linker = Linker()
        notes_updated = 0
        links_added = 0
        total = len(all_paths)

        from tqdm import tqdm

        for path in tqdm(all_paths, desc="● Finding links", file=sys.stderr, ncols=60):
            try:
                content = read_note(path)
            except ShardError:
                continue

            # Get titles excluding the current note
            current_title = None
            for t, p in title_map.items():
                if p == path:
                    current_title = t
                    break
            other_titles = [t for t in all_titles if t != current_title]

            suggestions = linker.find_links(content, other_titles)
            if not suggestions:
                continue

            new_content = apply_links(content, suggestions)
            if new_content == content:
                continue

            # Count actual substitutions by checking which suggestions matched
            actual_count = sum(
                1 for s in suggestions
                if s.linked_text in new_content
            )
            links_added += actual_count
            notes_updated += 1

            if verbose:
                for s in suggestions:
                    _err.print(
                        f"  [dim]{path.name}:[/dim] "
                        f"[cyan]{s.original_text}[/cyan] → "
                        f"[green]{s.linked_text}[/green]"
                    )

            if dry_run:
                continue

            try:
                path.write_text(new_content, encoding="utf-8")
            except OSError as exc:
                _err.print(f"[yellow]Warning:[/yellow] Could not write {path.name}: {exc}")

        _err.print("[green]●[/green] Writing changes...             [green]✓[/green]")

    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    _out.print()
    if dry_run:
        _out.print("[bold yellow]Dry run complete (no files changed)[/bold yellow]")
    else:
        _out.print("[bold green]✓ Sync complete[/bold green]")
    _out.print(f"  Notes updated:  {notes_updated}")
    _out.print(f"  Links added:    {links_added}")
    if not dry_run:
        _out.print(f"  Backup saved:   {backup_dir}")


# ── model ─────────────────────────────────────────────────────────────────────


@cli.group("model", invoke_without_command=True)
@click.pass_context
def model(ctx: click.Context) -> None:
    """Manage models and API keys.

    With no subcommand, shows the current model and available options.
    See also: shard config --set model=<model>
    """
    if ctx.invoked_subcommand is not None:
        return
    # Show current model status

    try:
        config = get_config()
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    _out.print()
    _out.print(f"  [bold]Current model:[/bold] {config.model or '[dim](not set)[/dim]'}")
    _out.print()
    _out.print("  [dim]Switch model:[/dim]    shard model use <model>")
    _out.print("  [dim]Pull model:[/dim]     shard model pull <model>")
    _out.print("  [dim]Add API key:[/dim]    shard model key <provider>")
    _out.print("  [dim]List all:[/dim]       shard model list")
    _out.print()


@model.command("list")
def model_list() -> None:
    """Show all available models grouped by tier."""
    from shard.models import list_models

    try:
        models = list_models()
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    tiers = {
        "local_small": ("Local Small", "Free, ~4GB RAM", "green"),
        "local_large": ("Local Large", "Free, 8GB+ RAM", "yellow"),
        "cloud": ("Cloud", "API key required", "blue"),
    }

    _out.print()
    for tier_key, (tier_name, tier_desc, color) in tiers.items():
        tier_models = [m for m in models if m["tier"] == tier_key]
        if not tier_models:
            continue

        _out.print(f"  [{color}]●[/{color}] [bold]{tier_name}[/bold] [dim]({tier_desc})[/dim]")
        for m in tier_models:
            prefix = "  ▶ " if m["current"] else "    "

            # Status indicators
            status_parts = []
            if m["current"]:
                status_parts.append("[bold green]current[/bold green]")

            if m["provider"] == "ollama":
                if m.get("pulled"):
                    prefix_mark = "✓ " if not m["current"] else "✓ "
                else:
                    prefix_mark = "  "
                    status_parts.append("[dim]not pulled[/dim]")
            else:
                prefix_mark = "  "
                if m.get("has_key"):
                    status_parts.append("[green]key set[/green]")
                else:
                    status_parts.append("[dim]no key[/dim]")
                if m.get("free"):
                    status_parts.append("[dim]free[/dim]")

            status = ", ".join(status_parts)
            status_str = f"  [{status}]" if status else ""

            _out.print(f"  {prefix}{prefix_mark}{m['name']:<35s}{status_str}")
        _out.print()

    _out.print("  [dim]To pull a local model:   shard model pull <name>[/dim]")
    _out.print("  [dim]To add an API key:       shard model key <provider>[/dim]")
    _out.print("  [dim]To switch model:         shard model use <name>[/dim]")
    _out.print()


@model.command("use")
@click.argument("model_name", metavar="MODEL")
def model_use(model_name: str) -> None:
    """Switch to a different model."""
    from shard.models import (
        MODEL_CATALOG,
        PROVIDER_ENV_MAP,
        PROVIDER_KEY_URLS,
        _detect_provider,
        detect_available_models,
        pull_ollama_model,
    )

    try:
        config = get_config()
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    # Check if it's an Ollama model
    is_ollama = model_name.startswith("ollama_chat/") or model_name.startswith("ollama/")

    if is_ollama:
        # Check if pulled
        detected = detect_available_models()
        if model_name not in detected:
            short_name = model_name.replace("ollama_chat/", "").replace("ollama/", "")
            if click.confirm(f"'{short_name}' is not pulled yet. Pull it now?", default=True):
                success = pull_ollama_model(short_name)
                if not success:
                    _err.print("[bold red]Pull failed.[/bold red] Try manually: "
                             f"ollama pull {short_name}")
                    sys.exit(1)
                _err.print(f"[green]✓[/green] Pulled {short_name}")
            else:
                _err.print(f"[yellow]Aborted.[/yellow] Pull manually: ollama pull {short_name}")
                sys.exit(0)
    else:
        # Check if cloud model needs an API key (catalog or heuristic)
        catalog_entry = next((e for e in MODEL_CATALOG if e["name"] == model_name), None)
        provider = catalog_entry["provider"] if catalog_entry else _detect_provider(model_name)

        if provider and provider in PROVIDER_ENV_MAP:
            env_var = PROVIDER_ENV_MAP[provider]
            has_key = bool(config.api_keys.get(provider)) or bool(os.environ.get(env_var))

            if not has_key:
                url = PROVIDER_KEY_URLS.get(provider, "")
                if click.confirm(
                    f"'{model_name}' requires a {provider} API key. Add it now?",
                    default=True,
                ):
                    key = click.prompt(f"  {provider} API key", hide_input=True)
                    if not key.strip():
                        _err.print("[bold red]Error:[/bold red] Key cannot be empty.")
                        sys.exit(1)
                    config.api_keys[provider] = key.strip()
                    os.environ[env_var] = key.strip()
                else:
                    _err.print(f"[yellow]Aborted.[/yellow] Add key with: shard model key {provider}")
                    sys.exit(0)

    config.model = model_name
    save_config(config)
    _err.print(f"[green]✓[/green] Switched to [bold]{model_name}[/bold]")


@model.command("pull")
@click.argument("model_name", metavar="MODEL")
def model_pull(model_name: str) -> None:
    """Pull an Ollama model."""
    from shard.models import pull_ollama_model

    # Strip ollama_chat/ prefix if user included it
    short_name = model_name.replace("ollama_chat/", "").replace("ollama/", "")

    _err.print(f"[dim]Pulling {short_name} from Ollama…[/dim]")
    success = pull_ollama_model(short_name)

    if not success:
        _err.print(f"[bold red]Failed to pull {short_name}.[/bold red]")
        sys.exit(1)

    _err.print(f"[green]✓[/green] Pulled {short_name}")

    full_name = f"ollama_chat/{short_name}"
    if click.confirm(f"Set {short_name} as your default model?", default=True):
        try:
            config = get_config()
            config.model = full_name
            save_config(config)
            _err.print(f"[green]✓[/green] Default model set to [bold]{full_name}[/bold]")
        except ShardError as exc:
            _err.print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(1)


@model.command("key")
@click.argument("provider", required=False, default=None)
@click.option("--list", "list_keys", is_flag=True, default=False, help="Show all configured API keys.")
@click.option("--remove", "remove_provider", default=None, metavar="PROVIDER", help="Remove a provider's API key.")
def model_key(provider: str | None, list_keys: bool, remove_provider: str | None) -> None:
    """Add or update an API key for a cloud provider."""
    from shard.models import KEY_PREFIX_HINTS, PROVIDER_ENV_MAP, PROVIDER_KEY_URLS

    try:
        config = get_config()
    except ShardError as exc:
        _err.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    # --list: show all keys with masking
    if list_keys:
        table = Table(show_header=False, expand=False, box=None, padding=(0, 2))
        table.add_column("Provider", style="bold")
        table.add_column("Key", style="dim")
        table.add_column("Status")

        for prov in sorted(PROVIDER_ENV_MAP.keys()):
            key = config.api_keys.get(prov, "")
            if key:
                masked = key[:6] + "..." + key[-5:] if len(key) > 20 else key[:3] + "..."
                table.add_row(prov, masked, "[green]✓[/green]")
            else:
                table.add_row(prov, "not set", "[dim]—[/dim]")

        _out.print()
        _out.print(table)
        _out.print()
        return

    # --remove: delete a key
    if remove_provider:
        env_var = PROVIDER_ENV_MAP.get(remove_provider)
        if remove_provider in config.api_keys:
            del config.api_keys[remove_provider]
            save_config(config)
            if env_var and env_var in os.environ:
                del os.environ[env_var]
            _err.print(f"[green]✓[/green] Removed {remove_provider} API key")
        else:
            _err.print(f"[yellow]No key configured for {remove_provider}.[/yellow]")
        return

    # Add/update key for a provider
    if provider is None:
        _err.print("[bold red]Error:[/bold red] Specify a provider: shard model key <provider>")
        _err.print(f"[dim]Available: {', '.join(sorted(PROVIDER_ENV_MAP.keys()))}[/dim]")
        sys.exit(1)

    if provider not in PROVIDER_ENV_MAP:
        _err.print(
            f"[bold red]Error:[/bold red] Unknown provider [bold]{provider}[/bold]. "
            f"Available: {', '.join(sorted(PROVIDER_ENV_MAP.keys()))}"
        )
        sys.exit(1)

    key = click.prompt(f"  {provider} API key", hide_input=True)
    key = key.strip()

    if not key:
        _err.print("[bold red]Error:[/bold red] Key cannot be empty.")
        sys.exit(1)

    # Prefix warning (not rejection)
    expected_prefix = KEY_PREFIX_HINTS.get(provider)
    if expected_prefix and not key.startswith(expected_prefix):
        _err.print(
            f"[yellow]Warning:[/yellow] This doesn't look like a standard {provider} key "
            f"(expected prefix: {expected_prefix}). Saving anyway."
        )

    config.api_keys[provider] = key
    save_config(config)

    env_var = PROVIDER_ENV_MAP[provider]
    os.environ[env_var] = key

    url = PROVIDER_KEY_URLS.get(provider, "")
    _err.print(f"[green]✓[/green] {provider.capitalize()} API key saved")
    if url:
        _err.print(f"  [dim]Get keys at: {url}[/dim]")


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
    "notes_subfolder",
}

_FIELD_HELP = {
    "vault_path": "Absolute path to your Obsidian vault directory.",
    "model": "LiteLLM model string used for note generation (e.g. ollama_chat/qwen2.5:3b).",
    "chroma_path": "Directory where ChromaDB persists its vector index.",
    "embedding_model": "Sentence-transformers model name for vector embeddings.",
    "notes_subfolder": "Vault-relative subfolder for new notes (empty = vault root).",
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

    See also: shard model (manage models and API keys)
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
    elif key == "notes_subfolder" and ".." in value:
        _err.print(
            "[bold red]Error:[/bold red] notes_subfolder must not contain '..' "
            "(path traversal is not allowed)."
        )
        sys.exit(1)
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
        ("notes_subfolder", cfg.notes_subfolder or "[dim](vault root)[/dim]"),
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
