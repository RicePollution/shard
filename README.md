# Shard — AI-Powered Note Ingestion for Obsidian

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-success.svg)](https://ollama.com)

Shard ingests PDFs, URLs, YouTube videos, and text into your Obsidian vault as structured AI-generated notes with semantic search. Everything runs locally by default — your data never leaves your machine.

<!-- TODO: Add demo GIF ![shard demo](assets/demo.gif) -->

## ✨ Features

- 📄 Multi-source ingestion (PDF, URL, YouTube, text, stdin)
- 🤖 AI-powered formatting with auto-generated titles, tags, summaries
- 🔍 Semantic search with ChromaDB vector embeddings
- 🏠 Local-first with Ollama — no cloud, no API keys, no cost
- 📝 Native Obsidian integration with YAML frontmatter
- 🔌 Flexible model support via LiteLLM (Ollama, OpenAI, Anthropic, Groq, etc.)
- ⚡ Fuzzy note search and one-click Obsidian opening
- 🧠 **Learn your style** — analyzes your vault and writes new notes that match how you already write
- ⚛️ **Atomic notes** — automatically splits any source into focused single-concept notes, all interlinked with [[wikilinks]]
- 🔗 **Auto backlinks** — syncs [[wikilinks]] across your vault to build a rich knowledge graph
- 📁 **Flat file saving** — notes save directly to your vault root, no buried subfolders

## 📋 Prerequisites

### 🐍 Python 3.11+

Python is the programming language Shard is built with. You need version 3.11 or newer.

Check if installed:

```bash
python3 --version
```

<details>
<summary>🐧 Arch Linux</summary>

```bash
sudo pacman -S python
```

</details>

<details>
<summary>🐧 Ubuntu / Debian</summary>

```bash
sudo apt update
sudo apt install python3
```

</details>

<details>
<summary>🐧 Fedora / RHEL</summary>

```bash
sudo dnf install python3
```

</details>

<details>
<summary>🍎 macOS</summary>

```bash
brew install python3
```

</details>

<details>
<summary>🪟 Windows</summary>

Download the installer from [python.org](https://www.python.org/downloads/) and run it. Make sure to check "Add Python to PATH" during installation.

</details>

Verify:

```bash
python3 --version  # Must show 3.11 or higher
```

### 📦 uv

uv is a modern Python package manager. It's faster than pip and lets you install Python apps as global commands without managing virtual environments. Think of it like installing an app, not a library.

Check if installed:

```bash
uv --version
```

<details>
<summary>🐧 Arch Linux</summary>

```bash
sudo pacman -S uv
```

</details>

<details>
<summary>🐧 Ubuntu / Debian</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

</details>

<details>
<summary>🐧 Fedora / RHEL</summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

</details>

<details>
<summary>🍎 macOS</summary>

```bash
brew install uv
```

Or via the installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
```

</details>

<details>
<summary>🪟 Windows</summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

</details>

Verify:

```bash
uv --version
```

### 🦙 Ollama

Ollama runs AI models locally on your machine for free. No account, no API key, no internet needed after setup. Shard uses it by default so your notes stay completely private.

Check if installed:

```bash
ollama --version
```

<details>
<summary>🐧 Arch Linux</summary>

```bash
sudo pacman -S ollama
```

</details>

<details>
<summary>🐧 Ubuntu / Debian</summary>

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

</details>

<details>
<summary>🐧 Fedora / RHEL</summary>

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

</details>

<details>
<summary>🍎 macOS</summary>

```bash
brew install ollama
```

Or download from [ollama.com](https://ollama.com).

</details>

<details>
<summary>🪟 Windows</summary>

Download the installer from [ollama.com](https://ollama.com) and run it.

</details>

**Start Ollama:**

<details>
<summary>🐧 Arch Linux</summary>

```bash
ollama serve
```

Or for automatic startup on boot (recommended):

```bash
sudo systemctl enable --now ollama
```

</details>

<details>
<summary>🐧 Ubuntu / Debian</summary>

```bash
ollama serve
```

Or for automatic startup on boot (recommended):

```bash
sudo systemctl enable --now ollama
```

</details>

<details>
<summary>🐧 Fedora / RHEL</summary>

```bash
ollama serve
```

Or for automatic startup on boot (recommended):

```bash
sudo systemctl enable --now ollama
```

</details>

<details>
<summary>🍎 macOS</summary>

Ollama runs automatically as a menu bar app after installation. No extra steps needed.

</details>

<details>
<summary>🪟 Windows</summary>

Ollama runs automatically as a system tray app after installation. No extra steps needed.

</details>

> 💡 `ollama serve` must be running whenever you use Shard. On Linux, using `systemctl` is the easiest option — it starts automatically on boot.

**Pull the default model:**

```bash
ollama pull qwen2.5:3b
```

> 💡 This downloads about 2 GB. It runs on any modern machine with no GPU required. Completely free, forever.

Verify:

```bash
ollama run qwen2.5:3b "say hello"
```

### 🗃️ Obsidian Vault

Obsidian is a note-taking app that stores notes as plain markdown files in a folder called a "vault." Shard saves generated notes directly into your vault.

<details>
<summary>🐧 Arch Linux</summary>

```bash
yay -S obsidian
```

Or download the AppImage from [obsidian.md](https://obsidian.md).

</details>

<details>
<summary>🐧 Ubuntu / Debian</summary>

Download the `.deb` file from [obsidian.md](https://obsidian.md) and install it:

```bash
sudo apt install ./Obsidian-*.deb
```

</details>

<details>
<summary>🐧 Fedora / RHEL</summary>

Download the `.rpm` file from [obsidian.md](https://obsidian.md) and install it:

```bash
sudo dnf install ./Obsidian-*.rpm
```

</details>

<details>
<summary>🍎 macOS</summary>

```bash
brew install --cask obsidian
```

Or download from [obsidian.md](https://obsidian.md).

</details>

<details>
<summary>🪟 Windows</summary>

Download the installer from [obsidian.md](https://obsidian.md) and run it.

</details>

Open Obsidian and create a vault if you haven't. Note the vault folder path — you'll need it during Shard setup. Example paths:

- Linux: `/home/yourname/Documents/MyVault`
- macOS: `/Users/yourname/Documents/MyVault`
- Windows: `C:\Users\yourname\Documents\MyVault`

## 🚀 Installation

```bash
git clone https://github.com/RicePollution/shard
cd shard
uv tool install .
```

> 💡 `uv tool install .` installs Shard as a global command from the local folder. You only need to do this once. After this, `shard` works from any directory.

**If `shard` is not found:**

<details>
<summary>🐧 Arch Linux</summary>

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

</details>

<details>
<summary>🐧 Ubuntu / Debian</summary>

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

</details>

<details>
<summary>🐧 Fedora / RHEL</summary>

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

</details>

<details>
<summary>🍎 macOS</summary>

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

</details>

<details>
<summary>🪟 Windows</summary>

1. Press `Win + X` and select "System"
2. Click "Advanced system settings" on the left
3. Click "Environment Variables" button
4. Under "User variables", click "New"
5. Variable name: `PATH`
6. Variable value: `%USERPROFILE%\.local\bin`
7. Click OK on all dialogs
8. Restart your terminal

</details>

Verify:

```bash
shard --help
```

## ⚙️ First Time Setup

Run:

```bash
shard config
```

Example wizard output:
  
Note: Default path requires a space for /Documents/Obsidian Vault

```text
Welcome to Shard! Let's get you set up.

Path to your Obsidian vault [/home/user/Documents/ObsidianVault]: /home/user/Documents/MyVault

Checking for a local Ollama installation…
Ollama is running. Found 1 installed model(s).

Installed models:
  qwen2.5:3b

'qwen2.5:3b' is already installed. Using it as the default model.

Config saved to /home/user/.config/shard/config.json
```

**Config file location:**

| OS | Path |
|---|---|
| 🐧 Linux | `~/.config/shard/config.json` |
| 🍎 macOS | `~/.config/shard/config.json` |
| 🪟 Windows | `%APPDATA%\shard\config.json` |

## 📖 Usage

### `shard add`

Add a note from any source:

```bash
# From a PDF
shard add /path/to/paper.pdf

# From a URL
shard add "https://example.com/interesting-article"

# From YouTube (extracts transcript automatically)
shard add "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# From plain text
shard add "The key insight about neural networks is..."

# From stdin
cat notes.txt | shard add -

# Single note (classic behavior, no splitting)
shard add "https://example.com/article" --single
```

Example output:

```text
✓ Extracted: url — Example Article Title
✓ Formatted: Example Article Title (5 tags)
✓ Saved: Imported/Shards/example-article-title.md
✓ Indexed: 12 chunks

Shard added successfully.
  Title:  Example Article Title
  Tags:   machine-learning, neural-networks, deep-learning, ai, tutorial
  Chunks: 12
  Path:   /home/user/Documents/MyVault/Imported/Shards/example-article-title.md
```

### `shard ask`

```bash
shard ask "What are the best practices for testing?"

# Get more context
shard ask "machine learning basics" --top-k 10
```

Example output:

```text
Based on your notes, here are the key testing best practices:

1. Write tests before implementation (TDD approach)
2. Keep unit tests focused on a single behavior
3. Use mocks sparingly — prefer integration tests where possible

Sources: "Software Testing Fundamentals" (relevance: 89.2%),
"Clean Code Notes" (relevance: 74.5%)

┌─────────────────────────────┬──────────┬───────────┐
│ Title                       │ Path     │ Relevance │
├─────────────────────────────┼──────────┼───────────┤
│ Software Testing Fundamentals│ Imported │    89.2%  │
│ Clean Code Notes            │ Imported │    74.5%  │
└─────────────────────────────┴──────────┴───────────┘
```

### `shard index`

```bash
shard index
```

Reindex all notes. Run after manual edits or to rebuild the search index.

### `shard learn`

Analyzes your existing notes to learn your exact writing style —
headings, tags, frontmatter, tone, structure. Future notes from
`shard add` will be written to match your vault natively.

```bash
shard learn                    # normal depth (default)
shard learn --depth quick      # 1 API call, fast
shard learn --depth deep       # entire vault, most accurate
shard learn --force            # re-analyze
shard learn --show             # print current style fingerprint
shard learn --template         # print blank note template
```

Example output:
```
📝 Your note fingerprint:
┌─────────────────────────────────────────────────────┐
│ 1. Always opens with a one-sentence TL;DR in bold   │
│ 2. Uses ## TL;DR, ## Notes, ## Links as headings    │
│ 3. Tags: #lowercase-hyphen, 3-5 per note            │
│ 4. Frontmatter: tags, date, source                  │
│ 5. Ends every note with ## Related                  │
└─────────────────────────────────────────────────────┘
Average note length: ~320 words
```

### `shard sync`

Scans your vault and adds [[wikilinks]] between related notes.
Makes your Obsidian graph view much more connected and useful.
Always creates a backup before making any changes.

```bash
shard sync                # sync all backlinks
shard sync --dry-run      # preview links without changing files
shard sync --verbose      # show each link as it's added
```

Example output:
```
✓ Sync complete
  Notes updated:  34
  Links added:    127
  Backup saved:   ~/.shard/backups/2024-01-15T14:32:00/
```

### `shard list`

```bash
shard list
```

Show table of all imported notes with title, tags, date, path.

### `shard open`

```bash
shard open "machine learning"
```

Fuzzy-matches note titles and opens the best match in Obsidian.

### `shard config`

```bash
# View current config
shard config

# Update a setting
shard config --set model="ollama_chat/llama3.1:8b"
shard config --set vault_path="/new/vault/path"

# Re-run setup wizard
shard config --setup
```

## 🤖 Models

### 🏷️ Model Tiers

| Tier | Models | Cost | RAM Required |
|------|--------|------|-------------|
| 🟢 Local Small | `qwen2.5:3b`, `phi3.5`, `llama3.2:3b` | Free | ~4 GB |
| 🟡 Local Large | `llama3.1:8b`, `qwen2.5:14b` | Free | 8+ GB |
| 🔵 Cloud | Claude, GPT-4o, Gemini, Groq | Paid / Free tier | N/A |

### Switching models

```bash
# Use a different Ollama model
ollama pull llama3.1:8b
shard config --set model="ollama_chat/llama3.1:8b"

# Use a cloud model
shard config --set model="groq/llama-3.1-8b-instant"
```

### ☁️ Cloud Providers

| Provider | Model String | Free Tier | API Key |
|----------|-------------|-----------|---------|
| Groq | `groq/llama-3.1-8b-instant` | ✅ Yes | [console.groq.com](https://console.groq.com) |
| OpenAI | `gpt-4o` | ❌ No | [platform.openai.com](https://platform.openai.com/api-keys) |
| Anthropic | `claude-sonnet-4-20250514` | ❌ No | [console.anthropic.com](https://console.anthropic.com) |
| Google | `gemini/gemini-pro` | ✅ Yes | [aistudio.google.com](https://aistudio.google.com/apikey) |

> 💡 **Groq** offers a generous free tier with fast inference. It's the best starting point if you want cloud models without paying.

**Setting API keys:**

<details>
<summary>🐧 Arch Linux</summary>

```bash
export GROQ_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Make permanent by adding to ~/.bashrc:
echo 'export GROQ_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

</details>

<details>
<summary>🐧 Ubuntu / Debian</summary>

```bash
export GROQ_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Make permanent by adding to ~/.bashrc:
echo 'export GROQ_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

</details>

<details>
<summary>🐧 Fedora / RHEL</summary>

```bash
export GROQ_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Make permanent by adding to ~/.bashrc:
echo 'export GROQ_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

</details>

<details>
<summary>🍎 macOS</summary>

```bash
export GROQ_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Make permanent by adding to ~/.zshrc:
echo 'export GROQ_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

</details>

<details>
<summary>🪟 Windows</summary>

```powershell
[System.Environment]::SetEnvironmentVariable("GROQ_API_KEY", "your-key-here", "User")
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "your-key-here", "User")
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "your-key-here", "User")
```

After running the above commands, restart your terminal for the changes to take effect.

</details>

## ⚙️ Configuration Reference

| Field | Default | Description |
|-------|---------|-------------|
| `vault_path` | *(set during setup)* | Absolute path to your Obsidian vault |
| `model` | `ollama_chat/qwen2.5:3b` | LiteLLM model string for note generation |
| `chroma_path` | `~/.local/share/shard/chroma` | Directory for ChromaDB vector index |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |
| `custom_models` | `[]` | User-registered model descriptors |
| `api_keys` | `{}` | Provider API keys (alternative to env vars) |
| `notes_subfolder` | `""` (vault root) | Where new notes are saved |
| `style_profile` | auto-managed | Path to learned style JSON |

## 🏗️ How It Works

```text
Your Input → Extractor → Formatter (AI) → Indexer → Obsidian Vault
```

- **Extractor**: Pulls text from PDFs (pdfplumber), URLs (httpx + BeautifulSoup), YouTube (transcript API), or stdin
- **Formatter**: Sends text to your LLM to generate a structured note with title, tags, summary, and markdown body
- **Indexer**: Chunks the note text, generates vector embeddings (sentence-transformers), and stores them in ChromaDB for semantic search
- **Vault**: Saves the note as a markdown file with YAML frontmatter in your Obsidian vault

## 🔧 Troubleshooting

<details>
<summary>❌ shard: command not found</summary>

Your PATH doesn't include the uv tool install directory.

**🐧 Linux:**

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**🍎 macOS:**

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**🪟 Windows:**

1. Press `Win + X` and select "System"
2. Click "Advanced system settings" on the left
3. Click "Environment Variables" button
4. Under "User variables", click "New"
5. Variable name: `PATH`
6. Variable value: `%USERPROFILE%\.local\bin`
7. Click OK on all dialogs
8. Restart your terminal

</details>

<details>
<summary>❌ Connection refused / Cannot reach Ollama</summary>

Ollama isn't running. Start it:

**🐧 Linux:**

```bash
ollama serve
```

Or for automatic startup:

```bash
sudo systemctl enable --now ollama
```

**🍎 macOS:**

Open the Ollama app from your Applications folder.

**🪟 Windows:**

Open the Ollama app from your Start Menu.

</details>

<details>
<summary>❌ Model not found</summary>

Pull the default model:

```bash
ollama pull qwen2.5:3b
```

</details>

<details>
<summary>❌ Vault path does not exist</summary>

Re-run setup:

```bash
shard config --setup
```

</details>

<details>
<summary>⏳ Slow on first run</summary>

Normal! First run loads the AI model into memory and downloads the embedding model (~80 MB). Subsequent runs are much faster.

</details>

<details>
<summary>❌ Permission denied (Linux)</summary>

If you get permission errors on Linux:

```bash
chmod +x ~/.local/bin/shard
```

</details>

<details>
<summary>❌ shard learn says not enough notes</summary>

shard learn needs at least 5 notes in your vault to analyze.
Add some notes first with `shard add`, then re-run `shard learn`.

</details>

<details>
<summary>❌ shard sync changed something it shouldn't have</summary>

shard sync always creates a backup before making changes.
Find your backup at `~/.shard/backups/` and restore from there.

To preview changes without modifying files, always use:
```bash
shard sync --dry-run
```

</details>

## 🤝 Contributing

We welcome contributions! For guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

**Good first issues:**

- Add support for EPUB files
- Add support for RSS feeds
- Improve chunking with semantic boundaries
- Add `shard export` command
- Better error messages for common failures

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
