"""LLM model management with litellm."""

from __future__ import annotations

import logging
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
        If the underlying litellm call fails for any reason.
    """
    if not model:
        model = get_config().model

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if model.startswith(OLLAMA_PREFIX):
        kwargs["api_base"] = OLLAMA_BASE_URL

    try:
        response = litellm.completion(**kwargs)
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
    """List all known models with availability status.

    Merges locally detected Ollama models with any custom models stored in the
    Shard configuration.  Each entry is a dict with keys:

    - ``name``  -- the model identifier
    - ``provider`` -- ``"ollama"`` for detected models, or the registered provider
    - ``available`` -- *True* if the model was detected as running locally or
      if the custom model has an API key configured
    """
    detected = set(detect_available_models())
    cfg = get_config()

    seen: set[str] = set()
    results: list[dict[str, Any]] = []

    # Ollama models detected on the local machine.
    for model_name in sorted(detected):
        seen.add(model_name)
        results.append({
            "name": model_name,
            "provider": "ollama",
            "available": True,
        })

    # Custom models from config (may overlap with detected Ollama models).
    for entry in cfg.custom_models:
        name = entry["name"]
        if name in seen:
            continue
        seen.add(name)
        provider = entry.get("provider", "unknown")
        has_key = bool(cfg.api_keys.get(provider))
        results.append({
            "name": name,
            "provider": provider,
            "available": has_key,
        })

    return results
