# Shard

Ingest PDFs, URLs, YouTube videos, and text into Obsidian as structured AI-generated notes with semantic search.

Shard automates your note-taking workflow: extract content from any source, generate AI-structured notes with tags and summaries, index them in ChromaDB for semantic search, and seamlessly integrate with your Obsidian vault.

## Features

- **Multi-source ingestion**: PDF files, web URLs, YouTube transcripts, and raw text
- **AI-powered formatting**: Automatic title, tag, and summary generation using LLMs
- **Semantic search**: ChromaDB integration with sentence-transformers embeddings
- **Local-first**: Ollama support for running models locally without cloud APIs
- **Obsidian native**: Saves notes as frontmatter-rich markdown in your vault
- **Flexible models**: LiteLLM provider support (Ollama, OpenAI, Claude, Anthropic, and more)

## Quick Start

### Install

With `uv` (recommended):
```bash
uv pip install shard-cli
```

With `pip`:
```bash
pip install shard-cli
```

### First Run

```bash
shard add "https://example.com/article"
```

On your first command, Shard will walk you through an interactive setup:
1. Path to your Obsidian vault
2. Detection of local Ollama installation (optional)
3. Model selection and download if needed

Configuration is saved to `~/.config/shard/config.json`.

## Usage

### shard add

Add a note from any source:

```bash
# From a PDF file
shard add /path/to/document.pdf

# From a web URL
shard add "https://example.com/article"

# From a YouTube video (automatically extracts transcript)
shard add "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# From plain text
shard add "Machine learning is a subset of artificial intelligence..."

# From stdin
echo "Some text to import" | shard add -
```

Shard auto-detects the source type and extracts content. It then generates a structured note with:
- AI-generated title
- Relevant tags
- Key summary
- Full body content
- Source attribution

The note is saved to `<vault>/Imported/Shards/` and immediately indexed.

### shard ask

Query your notes with semantic search:

```bash
shard ask "What are the best practices for testing?"

# Retrieve more context chunks
shard ask "machine learning basics" --top-k 10
```

Returns an AI-generated answer sourced from your indexed notes with a relevance table showing sources.

### shard index

Reindex all notes in your vault:

```bash
shard index
```

Reads all notes under `<vault>/Imported/Shards/`, chunks their content, and upserts chunks into ChromaDB. Run this after manually editing notes or recovering from a corrupted index.

### shard list

List all imported shard notes:

```bash
shard list
```

Displays a table with title, tags, creation date, and path for each note in your vault.

### shard open

Quickly open a note in Obsidian:

```bash
shard open "machine learning"
```

Fuzzy-matches note titles and opens the best match via the `obsidian://` URI scheme.

### shard config

View and manage configuration:

```bash
# Display current configuration
shard config

# Update a single setting
shard config --set vault_path="/path/to/vault"
shard config --set model="ollama_chat/llama3"
shard config --set embedding_model="all-MiniLM-L6-v2"

# Rerun interactive setup
shard config --setup
```

**Settable fields:**
- `vault_path`: Absolute path to your Obsidian vault directory
- `model`: LiteLLM model string for note generation (e.g., `ollama_chat/qwen2.5:3b`)
- `chroma_path`: Directory for ChromaDB persistence
- `embedding_model`: Sentence-transformers model name for embeddings

## Models

### Local Models (Recommended)

Shard defaults to **Ollama with qwen2.5:3b**, a fast 3B parameter model that runs locally without cloud APIs or API keys.

**Setup:**
1. [Install Ollama](https://ollama.ai)
2. Run `ollama serve` in a terminal
3. Run `shard add <source>` — Shard will auto-detect Ollama and offer to pull the model

**Using a different Ollama model:**
```bash
shard config --set model="ollama_chat/llama2"
shard config --set model="ollama_chat/neural-chat"
```

[Browse available Ollama models](https://ollama.ai/library)

### Cloud Providers

Use OpenAI, Anthropic, Azure, or other providers via LiteLLM. Set your API key as an environment variable:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
shard config --set model="gpt-4-turbo"

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."
shard config --set model="claude-3-sonnet"

# Azure OpenAI
export AZURE_API_KEY="..."
shard config --set model="azure/gpt-4"
```

See [LiteLLM provider documentation](https://docs.litellm.ai/docs/providers) for complete list and syntax.

## Configuration

Configuration is stored in `~/.config/shard/config.json` (created automatically on first run).

Example config:
```json
{
  "vault_path": "/home/user/Documents/ObsidianVault",
  "model": "ollama_chat/qwen2.5:3b",
  "chroma_path": "/home/user/.local/share/shard/chroma",
  "embedding_model": "all-MiniLM-L6-v2",
  "custom_models": [],
  "api_keys": {}
}
```

**Fields:**
- `vault_path` (required): Absolute path to Obsidian vault root
- `model`: LiteLLM model string for note generation (default: `ollama_chat/qwen2.5:3b`)
- `chroma_path`: Directory for semantic search embeddings (default: `~/.local/share/shard/chroma`)
- `embedding_model`: Sentence-transformers model (default: `all-MiniLM-L6-v2`)
- `custom_models`: Advanced model descriptors (see LiteLLM docs)
- `api_keys`: Provider API keys (alternative to environment variables)

## Requirements

- **Python 3.11+**
- **Ollama** (optional but recommended for local models)
  - [Download Ollama](https://ollama.ai)
  - Run `ollama serve` to start the daemon
  - Shard will auto-detect and use it if available

## License

MIT
