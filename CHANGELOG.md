# Changelog

All notable changes to shard are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com).
Versioning follows [Semantic Versioning](https://semver.org).

---

## [0.3.0] — 2026-03-11

### Breaking Changes
- **Redis Stack replaces ChromaDB** — `shard ask` and `shard index` now require Redis Stack with RediSearch. The `chroma_path` config field is no longer used. A one-time migration notice will appear on first run.

### Added
- Concurrent note generation — Stage B runs via `asyncio.gather` with Semaphore(3)
- Style matching overhaul — `shard learn` collects real vault excerpts and forbidden patterns for few-shot prompting
- Hard 3–5 subnote enforcement in Python after model response, with automatic retry
- `shard add --instruction/-i` flag for custom formatting directives
- Live model list — `shard model list` fetches from OpenAI/Anthropic APIs with 24h disk cache
- Hardcoded model lists for Groq, Gemini, and Mistral providers
- Redis connection pooling and HNSW vector index with cosine distance
- Async LLM completions via `litellm.acompletion`
- Redis Stack prerequisite and troubleshooting sections in README

### Changed
- Vector store backend from ChromaDB to Redis Stack (RediSearch + HNSW)
- Config fields: removed `chroma_path`, added `redis_host`, `redis_port`, `redis_password`

### Fixed
- Added `numpy` and `requests` as explicit dependencies (were transitive only)
- Removed unused `python-dotenv` dependency
- Removed dead `DEFAULT_CHROMA_PATH` constant
- Deduplicated Ollama model pull logic between config.py and models.py

---

## [0.2.1] — 2026-03-11

### Added
- Live status feed on shard add, ask, learn, sync, index showing current step and file being processed
- `StatusFeed` context manager in `shard/ui/status.py` with animated braille spinner

### Changed
- README: First Time Setup section moved above Commands for better onboarding flow

### Fixed
- JSON parse error when local models produce unescaped control characters in responses
- Completion timeout for local Ollama models increased to 600s (was 120s)

---

## [0.2.0] — 2026-03-10

### Added
- `shard model` command for interactive model and key management
- `shard model use`, `shard model pull`, `shard model key` subcommands
- `shard model list` shows all models grouped by tier with status indicators
- Atomic note splitting — `shard add` now generates multiple focused notes per source with automatic [[wikilinks]]
- `--single` flag on `shard add` to bypass splitting
- `shard learn --depth` flag with quick, normal, deep modes
- Friendly error messages when API keys are missing or invalid
- API keys auto-loaded from config.json into environment on model invocation
- Model catalog with tier/provider metadata for built-in model discovery

### Changed
- README reorganized for readability with Quick Start and Commands reference sections
- Prerequisites moved below Quick Start
- API key setup simplified to `shard model key <provider>` (replaces OS-specific env var instructions)

### Fixed
- API keys stored in config.json are now automatically passed to litellm

---

## [0.1.0] — 2026-03-09

### Added
- Initial release
- `shard add` — ingest PDF, URL, YouTube, text, and stdin
- `shard ask` — semantic search with AI-generated answers
- `shard index` — rebuild vector search index
- `shard list` — browse imported notes as a table
- `shard open` — fuzzy-match and open notes in Obsidian
- `shard config` — setup wizard and configuration management
- `shard learn` — vault style analysis with fingerprint output
- `shard sync` — automatic [[wikilink]] injection with backup
- Local-first with Ollama, cloud support via LiteLLM
- Vector search with sentence-transformers embeddings
- YAML frontmatter with structured metadata
