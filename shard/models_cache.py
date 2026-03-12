"""Live model list fetching with 24-hour disk cache."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 86_400  # 24 hours
_FETCH_TIMEOUT = 5.0

GROQ_MODELS: list[str] = [
    "groq/llama3-70b",
    "groq/llama3-8b",
    "groq/mixtral-8x7b-32768",
]

GEMINI_MODELS: list[str] = [
    "gemini/gemini-2.0-flash",
    "gemini/gemini-1.5-pro",
    "gemini/gemini-1.5-flash",
]

MISTRAL_MODELS: list[str] = [
    "mistral/mistral-large-latest",
    "mistral/mistral-small-latest",
]


def _cache_path() -> Path:
    return Path.home() / ".config" / "shard" / "models_cache.json"


def _read_cache() -> dict[str, Any] | None:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    fetched_at_raw = data.get("fetched_at")
    if not fetched_at_raw:
        return None

    try:
        fetched_at = datetime.fromisoformat(fetched_at_raw.replace("Z", "+00:00"))
    except ValueError:
        return None

    age = (datetime.now(tz=timezone.utc) - fetched_at).total_seconds()
    if age > _CACHE_TTL_SECONDS:
        return None

    return data


def _write_cache(data: dict[str, Any]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fetched_at": datetime.now(tz=timezone.utc).isoformat(), **data}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write models cache: %s", exc)


def _fetch_openai_models(api_key: str) -> list[str]:
    try:
        resp = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        models: list[dict[str, Any]] = resp.json().get("data", [])
        result: list[str] = []
        for m in models:
            mid = m.get("id", "")
            if not mid:
                continue
            # Keep only chat-capable models; skip embeddings, tts, whisper, dall-e
            skip_keywords = ("embedding", "whisper", "tts", "dall-e", "davinci", "babbage",
                             "curie", "ada", "moderation", "realtime", "audio", "transcribe",
                             "search", "similarity", "instruct")
            if any(kw in mid for kw in skip_keywords):
                continue
            if "gpt" in mid or mid.startswith(("o1", "o3", "o4")):
                result.append(mid)
        return sorted(result)
    except Exception as exc:
        logger.debug("Failed to fetch OpenAI models: %s", exc)
        return []


def _fetch_anthropic_models(api_key: str) -> list[str]:
    try:
        resp = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=_FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # Response is either {"data": [...]} or a top-level list
        items: list[dict[str, Any]] = data.get("data", data) if isinstance(data, dict) else data
        return sorted(m["id"] for m in items if isinstance(m, dict) and "id" in m)
    except Exception as exc:
        logger.debug("Failed to fetch Anthropic models: %s", exc)
        return []


def _resolve_key(provider: str) -> str:
    """Return the API key for *provider* from config or environment, or empty string."""
    from shard.config import get_config
    from shard.models import PROVIDER_ENV_MAP

    env_var = PROVIDER_ENV_MAP.get(provider, "")
    env_key = os.environ.get(env_var, "") if env_var else ""
    if env_key:
        return env_key

    try:
        cfg = get_config()
        return cfg.api_keys.get(provider, "")
    except Exception:
        return ""


def get_live_models() -> dict[str, list[str]]:
    """Get live model lists by provider with caching.

    Attempts to fetch live lists from OpenAI and Anthropic when valid API keys
    are present.  Results are cached on disk for 24 hours.  Falls back to the
    disk cache when a live fetch fails, and uses hardcoded lists for Groq,
    Gemini, and Mistral which do not expose a queryable public models API.

    Returns:
        A mapping of provider name to a list of model identifier strings, e.g.::

            {
                "openai": ["gpt-4o", "gpt-4o-mini", ...],
                "anthropic": ["claude-sonnet-4-20250514", ...],
                "groq": ["groq/llama3-70b", ...],
                "gemini": ["gemini/gemini-2.0-flash", ...],
                "mistral": ["mistral/mistral-large-latest", ...],
            }
    """
    cached = _read_cache()

    # If cache is fresh, use it directly — no network requests
    if cached:
        return {
            "openai": cached.get("openai", []),
            "anthropic": cached.get("anthropic", []),
            "groq": GROQ_MODELS,
            "gemini": GEMINI_MODELS,
            "mistral": MISTRAL_MODELS,
        }

    openai_key = _resolve_key("openai")
    anthropic_key = _resolve_key("anthropic")

    openai_models = _fetch_openai_models(openai_key) if openai_key else []
    anthropic_models = _fetch_anthropic_models(anthropic_key) if anthropic_key else []

    if openai_models or anthropic_models:
        _write_cache({"openai": openai_models, "anthropic": anthropic_models})

    return {
        "openai": openai_models,
        "anthropic": anthropic_models,
        "groq": GROQ_MODELS,
        "gemini": GEMINI_MODELS,
        "mistral": MISTRAL_MODELS,
    }
