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

from shard.pipeline import ConfigError

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIG_PATH: Path = Path.home() / ".config" / "shard" / "config.json"
DEFAULT_CHROMA_PATH: Path = Path.home() / ".local" / "share" / "shard" / "chroma"
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
        chroma_path: Directory where ChromaDB persists its data.
        embedding_model: Sentence-transformers model name for vector embeddings.
        custom_models: User-defined model descriptors forwarded to LiteLLM.
        api_keys: Mapping of provider name to API key (e.g. ``{"openai": "sk-…"}``).
    """

    vault_path: Path
    model: str = DEFAULT_MODEL
    chroma_path: Path = field(default_factory=lambda: DEFAULT_CHROMA_PATH)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    custom_models: list[dict[str, Any]] = field(default_factory=list)
    api_keys: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce strings to Path objects when deserialised from JSON.
        if not isinstance(self.vault_path, Path):
            self.vault_path = Path(self.vault_path)
        if not isinstance(self.chroma_path, Path):
            self.chroma_path = Path(self.chroma_path)


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
    raw["chroma_path"] = str(config.chroma_path)
    return raw


def _dict_to_config(data: dict[str, Any]) -> ShardConfig:
    """Deserialise *data* into a :class:`ShardConfig`.

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
        chroma_path=Path(data.get("chroma_path", str(DEFAULT_CHROMA_PATH))),
        embedding_model=data.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        custom_models=data.get("custom_models", []),
        api_keys=data.get("api_keys", {}),
    )


# ── Public API ────────────────────────────────────────────────────────────────


def load_config(path: Path = CONFIG_PATH) -> ShardConfig:
    """Read the JSON config file and return a :class:`ShardConfig`.

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
    """Pull *model_name* from the Ollama registry with a Rich progress spinner.

    The pull request is streamed; each newline-delimited JSON chunk is
    consumed and the spinner label is updated with the ``status`` field so the
    user sees live progress.

    Args:
        model_name: The model tag to pull (e.g. ``"qwen2.5:3b"``).

    Returns:
        ``True`` if the pull completed without an HTTP error, ``False``
        otherwise.
    """
    try:
        with _console.status(
            f"[bold cyan]Pulling {model_name} from Ollama…[/bold cyan]",
            spinner="dots",
        ) as spinner:
            with httpx.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name},
                timeout=None,  # pulls can take minutes
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk: dict[str, Any] = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    status_text = chunk.get("status", "")
                    if status_text:
                        spinner.update(
                            f"[bold cyan]{model_name}:[/bold cyan] {status_text}"
                        )
        _console.print(f"[green]Model '{model_name}' pulled successfully.[/green]")
        return True
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        _console.print(f"[red]Failed to pull model '{model_name}': {exc}[/red]")
        return False


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

    _console.print("\n[dim]Checking for a local Ollama installation…[/dim]")
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
    return config


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
