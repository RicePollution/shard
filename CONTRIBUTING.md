# Contributing to Shard

Thanks for your interest in contributing! This guide covers development setup, code style, testing, and how to extend Shard.

## Development Setup

Clone the repository and install dependencies with `uv`:

```bash
git clone https://github.com/yourusername/shard.git
cd shard
uv sync
```

This creates an isolated venv with all dependencies (including dev tools like pytest and ruff).

Activate the environment:
```bash
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

## Running Tests

```bash
uv run pytest
```

Run specific tests:
```bash
uv run pytest tests/test_extractor.py -v
uv run pytest tests/test_config.py::test_first_run_setup
```

Run with coverage:
```bash
uv run pytest --cov=shard
```

## Code Style

Shard uses `ruff` for linting and formatting. Check your code before committing:

```bash
uv run ruff check shard/
uv run ruff format shard/
```

Configuration is in `pyproject.toml`:
- Target: Python 3.11+
- Line length: 100 characters
- Rules: E (errors), F (pyflakes), I (import sorting)

## Architecture Overview

```
shard/
├── cli.py              # Click CLI entry point (commands: add, ask, index, list, open, config)
├── config.py           # Configuration management and first-run setup
├── models.py           # LiteLLM wrapper for chat completions
├── vault.py            # Obsidian vault I/O (saving notes, reading frontmatter)
├── runner.py           # Pipeline orchestration (extract → format → save → index)
├── search.py           # Semantic search with ChromaDB
└── pipeline/
    ├── __init__.py     # Error hierarchy, dataclasses, SourceType enum
    ├── extractor.py    # Content extraction (PDF, URL, YouTube, text, stdin)
    ├── formatter.py    # AI-powered note formatting (title, tags, summary)
    └── indexer.py      # Semantic chunking and ChromaDB indexing
```

**Data flow for `shard add`:**
1. **CLI** (cli.py) → calls `run_add_pipeline()`
2. **Extractor** (extractor.py) → detects input type (PDF/URL/YouTube/text) → returns `ExtractedContent`
3. **Formatter** (formatter.py) → generates title, tags, summary via LLM → returns `FormattedNote`
4. **Vault** (vault.py) → saves markdown with YAML frontmatter → returns `Path`
5. **Indexer** (indexer.py) → chunks note, embeds with sentence-transformers, upserts to ChromaDB → returns `IndexedNote`

**Data flow for `shard ask`:**
1. **CLI** → calls `search_ask()`
2. **Search** (search.py) → embeds query, retrieves top-k chunks from ChromaDB
3. **Models** (models.py) → generates answer via LLM with context
4. **CLI** → displays answer and sources table

## Adding a New Input Source Type

To add support for a new source (e.g., RSS feeds, podcasts, email):

### 1. Add the `SourceType`
In `/home/keegan/Documents/Projects/shard/shard/pipeline/__init__.py`:
```python
class SourceType(Enum):
    PDF = auto()
    URL = auto()
    YOUTUBE = auto()
    STDIN = auto()
    TEXT = auto()
    RSS = auto()  # New type
```

### 2. Implement the extractor function
In `/home/keegan/Documents/Projects/shard/shard/pipeline/extractor.py`, add:

```python
def _extract_rss(url: str) -> ExtractedContent:
    """Extract articles from an RSS feed.

    Args:
        url: RSS feed URL

    Returns:
        ExtractedContent with feed content

    Raises:
        ExtractionError: If the feed cannot be fetched or parsed
    """
    # Your implementation here
    return ExtractedContent(
        text=combined_text,
        source=url,
        source_type=SourceType.RSS,
        title=feed_title,
        metadata={"feed_url": url, "article_count": str(count)},
    )
```

### 3. Add detection logic
In the main `extract()` function, add a detection branch:
```python
def extract(input_str: str) -> ExtractedContent:
    stripped = input_str.strip()

    # ... existing checks ...

    # --- RSS feed ---
    if _looks_like_rss_url(stripped):
        return _extract_rss(stripped)

    # ... fallback to text ...
```

### 4. Add a detection helper (if needed)
```python
def _looks_like_rss_url(url: str) -> bool:
    """Check if URL points to an RSS feed."""
    return "feed" in url or "rss" in url or url.endswith(".xml")
```

### 5. Add tests
Create tests in `tests/test_extractor.py`:
```python
def test_extract_rss_valid():
    result = extract("https://example.com/feed.xml")
    assert result.source_type == SourceType.RSS
    assert result.title  # Should have a title
    assert result.text   # Should have extracted text
```

Dependencies for your extractor (e.g., `feedparser`) should be added to `pyproject.toml`.

## Adding a New Model Provider

Shard uses [LiteLLM](https://docs.litellm.ai/docs/providers) to support multiple LLM providers. To add a new provider:

### 1. Understand LiteLLM conventions
LiteLLM uses provider prefixes for model names:
- `gpt-4` = OpenAI
- `claude-3-sonnet` = Anthropic Claude
- `gemini-pro` = Google Gemini
- `ollama_chat/qwen2.5:3b` = Local Ollama

### 2. Update configuration documentation
In `cli.py` and `README.md`, add example usage:
```bash
# Groq
export GROQ_API_KEY="..."
shard config --set model="groq/mixtral-8x7b-32768"
```

### 3. Test integration
In your test, set the API key and verify the model string works:
```python
def test_complete_with_groq():
    config = ShardConfig(vault_path=Path("/tmp"))
    config.model = "groq/mixtral-8x7b-32768"

    # This will call litellm internally
    response = complete("Hello", model=config.model)
    assert response  # Check that a response was generated
```

**No code changes needed** — LiteLLM handles the routing automatically based on the model prefix. Just document it and ensure users know how to set their API keys as environment variables.

## Error Handling

Shard has a clean error hierarchy in `pipeline/__init__.py`:
```python
class ShardError(Exception):
    """Base error for all Shard operations."""

class ExtractionError(ShardError):
    """Failed to extract content from a source."""

class FormattingError(ShardError):
    """Failed to format extracted content into a note."""

class IndexingError(ShardError):
    """Failed to index a note in ChromaDB."""

class ConfigError(ShardError):
    """Configuration is missing or invalid."""

class ModelError(ShardError):
    """Model invocation failed."""

class VaultError(ShardError):
    """Vault I/O error."""
```

When adding new functionality, raise the appropriate exception subclass with a descriptive message:
```python
if not pdf_path.exists():
    raise ExtractionError(f"PDF file not found: {pdf_path}")
```

The CLI catches `ShardError` and displays it nicely to the user.

## Configuration and Testing

For local testing with a custom vault:
```bash
# Create a test vault directory
mkdir -p /tmp/test_vault/Imported/Shards

# Run tests (pytest fixtures handle temp directories)
uv run pytest tests/
```

For integration testing with real Ollama:
```bash
# Start Ollama in another terminal
ollama serve

# Run integration tests (marked with @pytest.mark.integration)
uv run pytest tests/ -m integration
```

## Pull Request Checklist

Before submitting:
- [ ] Tests pass: `uv run pytest`
- [ ] Code is formatted: `uv run ruff format shard/`
- [ ] Linting passes: `uv run ruff check shard/`
- [ ] Docstrings are present (especially public APIs)
- [ ] Error handling is clean (use appropriate `ShardError` subclass)
- [ ] README or docs updated if adding features

## Cutting a Release

When a significant batch of features is complete:

1. Update `version` in `pyproject.toml` (semver: major.minor.patch)
2. Add a new version block to `CHANGELOG.md` with date and changes
3. Commit: `"chore: bump version to X.Y.Z"`
4. Push to origin
5. Run: `gh release create vX.Y.Z --title "..." --notes "..."`

**Version guidelines:**
- Patch (0.0.x): bug fixes only
- Minor (0.x.0): new features, backwards compatible
- Major (x.0.0): breaking changes

## Questions?

Open an issue or discussion on the repository. We're here to help!
