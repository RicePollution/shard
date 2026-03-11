"""LLM model management with litellm."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
import litellm

from shard.config import get_config, save_config
from shard.pipeline import ModelError

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_PREFIX = "ollama_chat/"
_OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"
_OLLAMA_TIMEOUT = 2.0
_COMPLETION_TIMEOUT = 120  # seconds; prevents indefinite hangs on stalled LLM calls
_LOCAL_COMPLETION_TIMEOUT = 600  # seconds; local models need more time on consumer hardware

MODEL_CATALOG: list[dict[str, str]] = [
    # Local Small (Free, ~4GB RAM)
    {"name": "ollama_chat/qwen2.5:3b", "tier": "local_small",
     "provider": "ollama", "label": "Qwen 2.5 3B"},
    {"name": "ollama_chat/phi3.5", "tier": "local_small",
     "provider": "ollama", "label": "Phi 3.5"},
    {"name": "ollama_chat/llama3.2:3b", "tier": "local_small",
     "provider": "ollama", "label": "Llama 3.2 3B"},
    # Local Large (Free, 8GB+ RAM)
    {"name": "ollama_chat/llama3.1:8b", "tier": "local_large",
     "provider": "ollama", "label": "Llama 3.1 8B"},
    {"name": "ollama_chat/qwen2.5:14b", "tier": "local_large",
     "provider": "ollama", "label": "Qwen 2.5 14B"},
    # Cloud (API key required)
    {"name": "gpt-4o", "tier": "cloud",
     "provider": "openai", "label": "GPT-4o"},
    {"name": "gpt-5", "tier": "cloud",
     "provider": "openai", "label": "GPT-5"},
    {"name": "claude-sonnet-4-20250514", "tier": "cloud",
     "provider": "anthropic", "label": "Claude Sonnet 4"},
    {"name": "groq/llama3-70b", "tier": "cloud",
     "provider": "groq", "label": "Llama 3 70B (Groq)", "free": "true"},
    {"name": "gemini/gemini-2.0-flash", "tier": "cloud",
     "provider": "gemini", "label": "Gemini 2.0 Flash", "free": "true"},
]

PROVIDER_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

PROVIDER_KEY_URLS: dict[str, str] = {
    "openai": "platform.openai.com/api-keys",
    "anthropic": "console.anthropic.com",
    "groq": "console.groq.com",
    "gemini": "aistudio.google.com/apikey",
    "mistral": "console.mistral.ai",
}

KEY_PREFIX_HINTS: dict[str, str] = {
    "openai": "sk-",
    "anthropic": "sk-ant-",
    "groq": "gsk_",
}


def inject_api_keys() -> None:
    """Load API keys from config into environment for litellm."""
    cfg = get_config()
    for provider, env_var in PROVIDER_ENV_MAP.items():
        key = cfg.api_keys.get(provider)
        if key and env_var not in os.environ:
            os.environ[env_var] = key


def _detect_provider(model: str) -> str:
    """Guess the provider from a model name.

    Parameters
    ----------
    model:
        The litellm model identifier to inspect.

    Returns
    -------
    str
        A provider name such as ``"openai"``, ``"anthropic"``, ``"ollama"``,
        or ``"unknown"`` when no match is found.
    """
    if model.startswith("ollama"):
        return "ollama"
    for entry in MODEL_CATALOG:
        if entry["name"] == model:
            return entry["provider"]
    # Heuristic fallbacks
    if "gpt" in model or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    if "claude" in model:
        return "anthropic"
    if model.startswith("groq/"):
        return "groq"
    if model.startswith("gemini/"):
        return "gemini"
    return "unknown"


def pull_ollama_model(model_name: str) -> bool:
    """Pull model_name from Ollama with Rich progress spinner.

    Streams the pull request from the Ollama API, updating a Rich spinner
    with live status text from each newline-delimited JSON chunk.

    Parameters
    ----------
    model_name:
        The model tag to pull (e.g. ``"qwen2.5:3b"``).

    Returns
    -------
    bool
        ``True`` on success, ``False`` on failure.
    """
    from rich.console import Console

    _console = Console(stderr=True)

    try:
        with _console.status(
            f"[bold cyan]Pulling {model_name}…[/bold cyan]",
            spinner="dots",
        ) as spinner:
            with httpx.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name},
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    status_text = chunk.get("status", "")
                    if status_text:
                        spinner.update(f"[bold cyan]{model_name}:[/bold cyan] {status_text}")
        return True
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.error("Failed to pull model %s: %s", model_name, exc)
        return False


def complete(prompt: str, system: str = "", model: str = "") -> str:
    """Send a chat completion request via litellm and return the response text.

    Parameters
    ----------
    prompt:
        The user message to send.
    system:
        Optional system message prepended to the conversation.
    model:
        The litellm model identifier.  Falls back to the configured default
        when empty.

    Returns
    -------
    str
        The text content of the first completion choice.

    Raises
    ------
    ModelError
        If the underlying litellm call fails for any reason, including
        authentication errors when an API key is missing or invalid.
    """
    if not model:
        model = get_config().model

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    is_local = model.startswith(OLLAMA_PREFIX)
    timeout = _LOCAL_COMPLETION_TIMEOUT if is_local else _COMPLETION_TIMEOUT
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "timeout": timeout}
    if is_local:
        kwargs["api_base"] = OLLAMA_BASE_URL

    try:
        inject_api_keys()
        response = litellm.completion(**kwargs)
    except litellm.AuthenticationError as exc:
        provider = _detect_provider(model)
        url = PROVIDER_KEY_URLS.get(provider, "")
        msg = f"API key missing or invalid for {provider}"
        if url:
            msg += f"\n\n  Fix it with:\n    shard model key {provider}\n\n  Get a key at: {url}"
        raise ModelError(msg) from exc
    except Exception as exc:
        logger.error("litellm completion failed for model %s: %s", model, exc)
        raise ModelError(f"Model completion failed ({model}): {exc}") from exc

    try:
        content: str = response.choices[0].message.content  # type: ignore[union-attr]
    except (AttributeError, IndexError, TypeError) as exc:
        raise ModelError(f"Unexpected response structure from {model}") from exc

    return content or ""


def detect_available_models() -> list[str]:
    """Detect locally-running Ollama models.

    Pings the Ollama REST API at localhost:11434 and returns each discovered
    model name prefixed with ``ollama_chat/`` (the litellm convention for
    Ollama chat models).

    Returns an empty list when Ollama is unreachable or returns an unexpected
    payload.
    """
    try:
        resp = httpx.get(_OLLAMA_TAGS_ENDPOINT, timeout=_OLLAMA_TIMEOUT)
        resp.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException):
        logger.debug("Ollama is not available at %s", OLLAMA_BASE_URL)
        return []

    try:
        data = resp.json()
        models: list[dict[str, Any]] = data.get("models", [])
    except (ValueError, AttributeError):
        logger.debug("Unexpected response from Ollama tags endpoint")
        return []

    return [f"{OLLAMA_PREFIX}{m['name']}" for m in models if "name" in m]


def register_model(name: str, provider: str, api_key: str = "") -> None:
    """Register a custom model in the Shard configuration.

    The model is appended to ``config.custom_models`` and, if an API key is
    supplied, stored in ``config.api_keys``.  The updated configuration is
    persisted immediately.

    Parameters
    ----------
    name:
        A unique model identifier (e.g. ``gpt-4o``, ``claude-sonnet-4-20250514``).
    provider:
        The provider backend (e.g. ``openai``, ``anthropic``, ``ollama``).
    api_key:
        Optional API key for the provider.
    """
    cfg = get_config()

    entry = {"name": name, "provider": provider}

    # Avoid duplicate registrations.
    existing_names = {m["name"] for m in cfg.custom_models}
    if name in existing_names:
        cfg.custom_models = [m if m["name"] != name else entry for m in cfg.custom_models]
    else:
        cfg.custom_models.append(entry)

    if api_key:
        cfg.api_keys[provider] = api_key

    save_config(cfg)
    logger.info("Registered model %s (provider: %s)", name, provider)


def list_models() -> list[dict[str, Any]]:
    """List all known models with availability and config status.

    Merges the :data:`MODEL_CATALOG` with locally detected Ollama models and
    the API key status from the active config.  Each entry is a dict with keys:

    - ``name`` -- the litellm model identifier
    - ``label`` -- human-readable display name
    - ``tier`` -- ``"local_small"``, ``"local_large"``, or ``"cloud"``
    - ``provider`` -- provider backend string
    - ``current`` -- ``True`` if this is the active model in config
    - ``pulled`` -- ``True``/``False`` for Ollama models, ``None`` for cloud
    - ``has_key`` -- ``True``/``False`` for cloud models, ``None`` for Ollama
    - ``free`` -- ``True`` if the model is known to be free to use

    Returns
    -------
    list[dict[str, Any]]
        One dict per known model, catalog entries first, then any additional
        Ollama models detected locally that are not in the catalog.
    """
    detected = set(detect_available_models())
    cfg = get_config()
    current_model = cfg.model

    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for entry in MODEL_CATALOG:
        name = entry["name"]
        provider = entry["provider"]
        seen.add(name)

        is_ollama = provider == "ollama"
        pulled = name in detected
        has_key = bool(cfg.api_keys.get(provider)) or bool(
            os.environ.get(PROVIDER_ENV_MAP.get(provider, ""))
        )

        results.append({
            "name": name,
            "label": entry.get("label", name),
            "tier": entry["tier"],
            "provider": provider,
            "current": name == current_model,
            "pulled": pulled if is_ollama else None,
            "has_key": has_key if not is_ollama else None,
            "free": entry.get("free") == "true",
        })

    # Add any detected Ollama models not in catalog
    for model_name in sorted(detected):
        if model_name not in seen:
            results.append({
                "name": model_name,
                "label": model_name.replace("ollama_chat/", ""),
                "tier": "local_small",
                "provider": "ollama",
                "current": model_name == current_model,
                "pulled": True,
                "has_key": None,
                "free": False,
            })

    return results
