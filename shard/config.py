"""Shard configuration management.

Handles reading, writing, and interactive first-run setup of the Shard CLI
configuration stored at ~/.config/shard/config.json.
"""

from __future__ import annotations

import functools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import click
import httpx
from rich.console import Console
from rich.panel import Panel

from shard.pipeline import ConfigError

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIG_PATH: Path = Path.home() / ".config" / "shard" / "config.json"

DEFAULT_MODEL: str = "ollama_chat/qwen2.5:3b"
DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_TIMEOUT: float = 2.0
PREFERRED_MODEL: str = "qwen2.5:3b"

_console = Console(stderr=True)


# ── Dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class ShardConfig:
    """Top-level configuration for the Shard CLI.

    Attributes:
        vault_path: Absolute path to the Obsidian vault root directory.
        model: LiteLLM model string used for note generation.
        embedding_model: Sentence-transformers model name for vector embeddings.
        redis_host: Hostname for the Redis Stack instance.
        redis_port: Port for the Redis Stack instance.
        redis_password: Optional password for the Redis Stack instance.
        custom_models: User-defined model descriptors forwarded to LiteLLM.
        api_keys: Mapping of provider name to API key (e.g. ``{"openai": "sk-…"}``).
        notes_subfolder: Vault-relative subdirectory where notes are saved.
            An empty string (the default) saves notes directly in the vault root.
    """

    vault_path: Path
    model: str = DEFAULT_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    custom_models: list[dict[str, Any]] = field(default_factory=list)
    api_keys: dict[str, str] = field(default_factory=dict)
    notes_subfolder: str = ""

    def __post_init__(self) -> None:
        # Coerce strings to Path objects when deserialised from JSON.
        if not isinstance(self.vault_path, Path):
            self.vault_path = Path(self.vault_path)


# ── Serialisation helpers ─────────────────────────────────────────────────────


def _config_to_dict(config: ShardConfig) -> dict[str, Any]:
    """Serialise *config* to a JSON-compatible dictionary.

    Path objects are stored as their string representation so they survive a
    round-trip through JSON without losing platform semantics.

    Args:
        config: The :class:`ShardConfig` instance to serialise.

    Returns:
        A plain dictionary suitable for ``json.dump``.
    """
    raw = asdict(config)
    raw["vault_path"] = str(config.vault_path)
    return raw


def _dict_to_config(data: dict[str, Any]) -> ShardConfig:
    """Deserialise *data* into a :class:`ShardConfig`.

    Unknown keys (including the legacy ``chroma_path`` field) are silently
    ignored; migration detection is handled separately by
    :func:`detect_chroma_migration`.

    Args:
        data: Dictionary loaded from the JSON config file.

    Returns:
        A fully populated :class:`ShardConfig` instance.

    Raises:
        ConfigError: If *vault_path* is missing from *data*.
    """
    if "vault_path" not in data:
        raise ConfigError("Config file is missing required field 'vault_path'.")
    return ShardConfig(
        vault_path=Path(data["vault_path"]),
        model=data.get("model", DEFAULT_MODEL),
        embedding_model=data.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        redis_host=data.get("redis_host", "localhost"),
        redis_port=int(data.get("redis_port", 6379)),
        redis_password=data.get("redis_password", ""),
        custom_models=data.get("custom_models", []),
        api_keys=data.get("api_keys", {}),
        notes_subfolder=data.get("notes_subfolder", ""),
    )


# ── Public API ────────────────────────────────────────────────────────────────


def load_config(path: Path = CONFIG_PATH) -> ShardConfig:
    """Read the JSON config file and return a :class:`ShardConfig`.

    Also runs :func:`detect_chroma_migration` to inform users who are
    upgrading from a ChromaDB-based installation.

    Args:
        path: Override the default config file path (useful in tests).

    Returns:
        A :class:`ShardConfig` populated from the file on disk.

    Raises:
        ConfigError: If the file does not exist, is not valid JSON, or is
            missing required fields.
    """
    if not path.exists():
        raise ConfigError(
            f"Config file not found at {path}. Run 'shard init' to create it."
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc}") from exc

    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file {path} contains invalid JSON: {exc}") from exc

    detect_chroma_migration(data)
    return _dict_to_config(data)


def save_config(config: ShardConfig, path: Path = CONFIG_PATH) -> None:
    """Write *config* to the JSON config file, creating parent directories.

    Args:
        config: The :class:`ShardConfig` instance to persist.
        path: Override the default config file path (useful in tests).

    Raises:
        ConfigError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(_config_to_dict(config), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        path.chmod(0o600)
    except OSError as exc:
        raise ConfigError(f"Cannot write config file {path}: {exc}") from exc

    # Bust the singleton cache whenever the config is explicitly saved so
    # subsequent calls to get_config() pick up the new values.
    get_config.cache_clear()


@functools.lru_cache(maxsize=1)
def get_config() -> ShardConfig:
    """Return the cached singleton :class:`ShardConfig`.

    The result is memoised after the first call.  Call
    ``get_config.cache_clear()`` (done automatically by :func:`save_config`)
    to force a re-read on the next access.

    Returns:
        The loaded :class:`ShardConfig`.

    Raises:
        ConfigError: Propagated from :func:`load_config` if the file is absent
            or malformed.
    """
    return load_config()


# ── ChromaDB migration detection ──────────────────────────────────────────────


_chroma_migration_shown = False


def detect_chroma_migration(data: dict[str, Any]) -> None:
    """Print a one-time migration notice when a legacy ``chroma_path`` key is found.

    This is called automatically by :func:`load_config` every time the config
    is read, so users upgrading from an older Shard version will see the notice
    on their next command without any active intervention required.

    Args:
        data: The raw dictionary loaded from config.json before deserialisation.
    """
    global _chroma_migration_shown
    chroma_path = data.get("chroma_path")
    if not chroma_path or _chroma_migration_shown:
        return
    _chroma_migration_shown = True

    _console.print(
        Panel(
            f"  ChromaDB index detected at [bold]{chroma_path}[/bold]\n\n"
            "  Shard now uses [bold]Redis Stack[/bold] for vector search.\n\n"
            "  Run [bold cyan]shard index[/bold cyan] to rebuild your search "
            "index in Redis.\n"
            "  You can safely delete the ChromaDB directory after reindexing.",
            title="[yellow]Migration Notice[/yellow]",
            expand=False,
        )
    )


# ── Redis health check ────────────────────────────────────────────────────────


def check_redis(config: ShardConfig | None = None) -> bool:
    """Verify that Redis Stack is reachable and has the RediSearch module loaded.

    Attempts a ``PING`` to the configured Redis host/port.  If Redis is up,
    also checks that the ``RediSearch`` (FT) module is available by calling
    ``FT._LIST``.

    Prints a Rich panel with distro-specific install instructions on failure so
    users get actionable guidance without having to consult documentation.

    Args:
        config: Runtime config supplying host/port/password.  When ``None``,
            falls back to ``localhost:6379`` with no password.

    Returns:
        ``True`` if both Redis and RediSearch are available; ``False`` otherwise.
    """
    import redis as redis_lib  # local import — redis is an optional peer dep

    host = config.redis_host if config else "localhost"
    port = config.redis_port if config else 6379
    password = (config.redis_password or None) if config else None

    try:
        client = redis_lib.Redis(
            host=host,
            port=port,
            password=password,
            socket_connect_timeout=2,
            decode_responses=False,
        )
        client.ping()
    except Exception as exc:
        _console.print(
            Panel(
                f"  Could not connect to Redis at [bold]{host}:{port}[/bold].\n"
                f"  Error: [dim]{exc}[/dim]\n\n"
                "  [bold]Install Redis Stack:[/bold]\n\n"
                "  [cyan]Arch Linux[/cyan]\n"
                "    paru -S redis-stack  [dim]# or yay -S redis-stack[/dim]\n\n"
                "  [cyan]Ubuntu / Debian[/cyan]\n"
                "    curl -fsSL https://packages.redis.io/gpg | "
                "sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg\n"
                "    echo \"deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] "
                "https://packages.redis.io/deb $(lsb_release -cs) main\" "
                "| sudo tee /etc/apt/sources.list.d/redis.list\n"
                "    sudo apt update && sudo apt install redis-stack-server\n\n"
                "  [cyan]Fedora / RHEL[/cyan]\n"
                "    sudo dnf install redis-stack-server\n\n"
                "  [cyan]macOS[/cyan]\n"
                "    brew tap redis-stack/redis-stack\n"
                "    brew install redis-stack\n\n"
                "  Then start the service:\n"
                "    [bold]sudo systemctl enable --now redis-stack-server[/bold]  "
                "[dim](Linux)[/dim]\n"
                "    [bold]brew services start redis-stack[/bold]  [dim](macOS)[/dim]",
                title="[bold red]Redis Not Available[/bold red]",
                expand=False,
            )
        )
        return False

    # Redis is reachable — verify RediSearch module.
    try:
        client.execute_command("FT._LIST")
    except Exception as exc:
        _console.print(
            Panel(
                f"  Redis is running at [bold]{host}:{port}[/bold] but the "
                "[bold]RediSearch[/bold] module is not loaded.\n"
                f"  Error: [dim]{exc}[/dim]\n\n"
                "  Shard requires [bold]Redis Stack[/bold] (not plain Redis) for "
                "vector search.\n"
                "  Please replace your Redis installation with Redis Stack and "
                "restart the service.\n\n"
                "  [dim]See install instructions: https://redis.io/docs/stack/[/dim]",
                title="[bold red]RediSearch Module Missing[/bold red]",
                expand=False,
            )
        )
        return False

    return True


# ── Ollama helpers ────────────────────────────────────────────────────────────


def _fetch_ollama_models() -> list[str] | None:
    """Query the local Ollama daemon for installed model names.

    Returns:
        A sorted list of model name strings, or ``None`` if Ollama is
        unreachable within :data:`OLLAMA_TIMEOUT` seconds.
    """
    try:
        response = httpx.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        models: list[str] = [m["name"] for m in payload.get("models", [])]
        return sorted(models)
    except (httpx.RequestError, httpx.HTTPStatusError, KeyError, ValueError):
        return None


def _pull_ollama_model(model_name: str) -> bool:
    """Pull *model_name* via :func:`shard.models.pull_ollama_model`."""
    from shard.models import pull_ollama_model

    return pull_ollama_model(model_name)


# ── Interactive setup ─────────────────────────────────────────────────────────


def first_run_setup() -> ShardConfig:
    """Run an interactive first-run wizard and persist the resulting config.

    Workflow:
        1. Prompt the user for their Obsidian vault path and validate it.
        2. Check whether a local Ollama daemon is reachable.
        3. If reachable, display available models; offer to pull the preferred
           model if it is not already installed.
        4. If not reachable, warn the user and leave ``model`` empty so the
           caller can configure an alternative provider.
        5. Save and return the final :class:`ShardConfig`.

    Returns:
        The freshly created and saved :class:`ShardConfig`.

    Raises:
        click.Abort: If the user interrupts the wizard (Ctrl-C).
        ConfigError: If the config cannot be written to disk.
    """
    _console.print("\n[bold]Welcome to Shard![/bold] Let's get you set up.\n")

    # ── Step 1: vault path ────────────────────────────────────────────────────

    vault_path: Path = _prompt_vault_path()

    # ── Step 2: probe Ollama ──────────────────────────────────────────────────

    _console.print("\n[dim]Checking for a local Ollama installation...[/dim]")
    available_models = _fetch_ollama_models()

    chosen_model: str

    if available_models is None:
        _console.print(
            "[yellow]Warning:[/yellow] Could not reach Ollama at "
            f"{OLLAMA_BASE_URL}. You can set the model manually in "
            f"{CONFIG_PATH} once Ollama is running."
        )
        chosen_model = ""
    else:
        _console.print(
            f"[green]Ollama is running.[/green] "
            f"Found {len(available_models)} installed model(s)."
        )

        # ── Step 3: show models and optionally pull preferred one ─────────────

        if available_models:
            _console.print("\nInstalled models:")
            for name in available_models:
                _console.print(f"  [cyan]{name}[/cyan]")

        preferred_installed = any(
            m == PREFERRED_MODEL or m.startswith(f"{PREFERRED_MODEL}:")
            for m in available_models
        )

        if preferred_installed:
            chosen_model = DEFAULT_MODEL
            _console.print(
                f"\n[green]'{PREFERRED_MODEL}' is already installed.[/green] "
                "Using it as the default model."
            )
        else:
            _console.print(
                f"\n[yellow]'{PREFERRED_MODEL}' is not installed.[/yellow]"
            )
            should_pull: bool = click.confirm(
                f"Pull '{PREFERRED_MODEL}' now? (~2 GB download)",
                default=True,
            )
            if should_pull:
                success = _pull_ollama_model(PREFERRED_MODEL)
                chosen_model = DEFAULT_MODEL if success else ""
            else:
                # Let the user pick from already-installed models if any exist.
                if available_models:
                    _console.print(
                        "\nAvailable models to use instead:"
                    )
                    for idx, name in enumerate(available_models, start=1):
                        _console.print(f"  [{idx}] {name}")
                    chosen_model = click.prompt(
                        "Enter the model name to use (or leave blank to configure later)",
                        default="",
                        show_default=False,
                    ).strip()
                    # Prefix with LiteLLM provider if the user typed a bare name.
                    if chosen_model and not chosen_model.startswith("ollama"):
                        chosen_model = f"ollama_chat/{chosen_model}"
                else:
                    chosen_model = ""

    # ── Step 4: build, save and return config ─────────────────────────────────

    config = ShardConfig(
        vault_path=vault_path,
        model=chosen_model,
    )
    save_config(config)
    _console.print(f"\n[bold green]Config saved to {CONFIG_PATH}[/bold green]\n")

    # ── Step 5: offer learn and sync ──────────────────────────────────────────

    _offer_learn_and_sync(config)

    return config


# ── Post-setup helpers ────────────────────────────────────────────────────────


def _offer_learn_and_sync(config: ShardConfig) -> None:
    """Optionally run learn and sync after setup completes.

    Skips both prompts if the vault has fewer than 5 notes.  Failures are
    caught gracefully and never block the rest of setup.
    """
    from shard.vault import walk_vault

    try:
        note_count = len(walk_vault(config))
    except Exception:
        note_count = 0

    if note_count < 5:
        _console.print(
            "[dim]Add some notes first, then run "
            "[bold]shard learn[/bold] and [bold]shard sync[/bold][/dim]"
        )
        return

    # ── Prompt 1: shard learn ────────────────────────────────────────────────

    _console.print(
        "\n-> Would you like shard to [bold]learn your note style[/bold] now?\n"
        "  Analyzes your vault so new notes match your writing style.\n"
        f"  Recommended if you already have notes ({note_count} found)."
    )
    if click.confirm("  Run shard learn?", default=True):
        try:
            from pathlib import Path as _Path

            from shard.pipeline.learner import Learner, save_style_profile
            from shard.vault import read_note

            style_path = _Path.home() / ".shard" / "style.json"
            notes = []
            for p in walk_vault(config):
                try:
                    notes.append(read_note(p))
                except Exception:
                    continue

            learner = Learner()
            with _console.status(
                "[bold cyan]Learning your style...[/bold cyan]", spinner="dots"
            ):
                profile = learner.analyze(notes)
            save_style_profile(profile, style_path)
            _console.print("[green]Style profile saved[/green]")
        except Exception as exc:
            _console.print(
                f"[yellow]Could not run shard learn — try manually: "
                f"shard learn[/yellow] ({exc})"
            )
    else:
        _console.print("[dim]You can run this anytime with: shard learn[/dim]")

    # ── Prompt 2: shard sync ─────────────────────────────────────────────────

    _console.print(
        "\n-> Would you like to [bold]sync backlinks[/bold] across your vault now?\n"
        "  Adds [[wikilinks]] between related notes for the graph view.\n"
        f"  Recommended if you have 10+ notes ({note_count} found)."
    )
    if click.confirm("  Run shard sync?", default=True):
        try:
            import shutil
            from datetime import datetime, timezone
            from pathlib import Path as _Path

            from shard.pipeline.linker import Linker, apply_links
            from shard.vault import parse_frontmatter, read_note

            # Create backup
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            backup_dir = _Path.home() / ".shard" / "backups" / timestamp
            backup_dir.mkdir(parents=True, exist_ok=True)
            all_paths = walk_vault(config)
            for md_file in all_paths:
                rel = md_file.relative_to(config.vault_path)
                dest = backup_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(md_file, dest)

            # Build title index
            title_map: dict[str, Any] = {}
            for p in all_paths:
                try:
                    content = read_note(p)
                    metadata, _ = parse_frontmatter(content)
                    title_map[metadata.get("title", p.stem)] = p
                except Exception:
                    title_map[p.stem] = p

            all_titles = list(title_map.keys())
            linker = Linker()
            notes_updated = 0
            links_added = 0

            with _console.status(
                "[bold cyan]Syncing backlinks...[/bold cyan]", spinner="dots"
            ):
                for path in all_paths:
                    try:
                        content = read_note(path)
                    except Exception:
                        continue

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

                    links_added += len(suggestions)
                    notes_updated += 1
                    path.write_text(new_content, encoding="utf-8")

            _console.print(
                f"[green]Backlinks synced[/green] "
                f"({notes_updated} notes, {links_added} links)"
            )
        except Exception as exc:
            _console.print(
                f"[yellow]Could not run shard sync — try manually: "
                f"shard sync[/yellow] ({exc})"
            )
    else:
        _console.print("[dim]You can run this anytime with: shard sync[/dim]")


# ── Internal prompt helper ────────────────────────────────────────────────────


def _prompt_vault_path() -> Path:
    """Prompt the user for their Obsidian vault path, validating it exists.

    Loops until a valid, existing directory is entered.

    Returns:
        The resolved :class:`Path` to the vault directory.
    """
    while True:
        raw: str = click.prompt(
            "Path to your Obsidian vault",
            default=str(Path.home() / "Documents" / "ObsidianVault"),
        )
        candidate = Path(raw).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        _console.print(
            f"[red]'{candidate}' does not exist or is not a directory.[/red] "
            "Please try again."
        )
