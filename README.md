# Shard — AI-Powered Note Ingestion for Obsidian

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-success.svg)](https://ollama.com)

Shard ingests PDFs, URLs, YouTube videos, and text into your Obsidian vault as structured AI-generated notes with semantic search. Everything runs locally by default — your data never leaves your machine.

<!-- TODO: Add demo GIF ![shard demo](assets/demo.gif) -->

---

## ✨ Features

- 📄 Multi-source ingestion (PDF, URL, YouTube, text, stdin)
- 🤖 AI-powered formatting with auto-generated titles, tags, summaries
- 🔍 Semantic search with Redis Stack vector embeddings
- 🏠 Local-first with Ollama — no cloud, no API keys, no cost
- 📝 Native Obsidian integration with YAML frontmatter
- 🔌 Flexible model support via LiteLLM (Ollama, OpenAI, Anthropic, Groq, etc.)
- ⚡ Fuzzy note search and one-click Obsidian opening
- 🧠 **Learn your style** — analyzes your vault and writes new notes that match how you already write
- ⚛️ **Atomic notes** — automatically splits any source into focused single-concept notes, all interlinked with [[wikilinks]]
- 🔗 **Auto backlinks** — syncs [[wikilinks]] across your vault to build a rich knowledge graph
- 📁 **Flat file saving** — notes save directly to your vault root, no buried subfolders
- 🤖 **Model switcher** — switch between local and cloud models, pull Ollama models, and manage API keys with `shard model`

---

## ⚡ Quick Start

```bash
git clone https://github.com/RicePollution/shard
cd shard
uv tool install .
shard config
shard add "https://example.com/article"
shard ask "what did I just read?"
```

> Need Python, uv, or Ollama? See [Prerequisites](#-prerequisites).

---

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

---

## 📖 Commands

### shard add
Ingest any source into your vault as atomic notes.

| Flag | Description |
|---|---|
| `--single` | One note, no splitting |
| `--instruction`, `-i` | Custom formatting instruction (e.g. "focus on code examples") |

```bash
shard add "https://example.com"
shard add /path/to/paper.pdf
shard add "https://youtube.com/watch?v=..."
shard add "raw text or idea"
cat file.txt | shard add -
shard add "https://example.com" -i "focus on practical examples"
```

### shard ask
Semantic search across your entire vault with AI answers.

| Flag | Description |
|---|---|
| `--top-k N` | Number of source chunks to retrieve (default: 5) |

```bash
shard ask "what do I know about RAG?"
shard ask "best practices for testing" --top-k 10
```

### shard model
Manage models and API keys.

| Subcommand | Description |
|---|---|
| `shard model` | Interactive model menu |
| `shard model list` | Show all available models |
| `shard model use <model>` | Switch to a model |
| `shard model pull <model>` | Pull an Ollama model |
| `shard model key <provider>` | Add a cloud API key |
| `shard model key --list` | Show configured keys |

```bash
shard model use gpt-5
shard model pull llama3.1:8b
shard model key openai
```

### shard learn
Learn your vault's writing style for better note generation.

| Flag | Description |
|---|---|
| `--depth quick` | 5 notes, 1 API call, fastest |
| `--depth normal` | 20 notes, balanced (default) |
| `--depth deep` | Entire vault, most accurate |
| `--force` | Re-analyze even if profile exists |
| `--show` | Print current style fingerprint |
| `--template` | Print blank note template |

```bash
shard learn
shard learn --depth deep
```

### shard sync
Add [[wikilinks]] between related notes automatically.

| Flag | Description |
|---|---|
| `--dry-run` | Preview changes without modifying files |
| `--verbose` | Show each link as it's added |

```bash
shard sync
shard sync --dry-run
```

### shard index
Rebuild the semantic search index from your vault.

```bash
shard index
```

### shard list
Show all imported notes.

```bash
shard list
```

### shard open
Fuzzy-match and open a note in Obsidian.

```bash
shard open "machine learning"
```

### shard config
View and update settings.

| Flag | Description |
|---|---|
| `--show` | Print current config |
| `--set KEY=VALUE` | Update a single value |
| `--setup` | Re-run setup wizard |

```bash
shard config --show
shard config --set vault_path="/new/path"
```

---

## 🤖 Models

### 🏷️ Model Tiers

| Tier | Models | Cost | RAM Required |
|------|--------|------|-------------|
| 🟢 Local Small | `qwen2.5:3b`, `phi3.5`, `llama3.2:3b` | Free | ~4 GB |
| 🟡 Local Large | `llama3.1:8b`, `qwen2.5:14b` | Free | 8+ GB |
| 🔵 Cloud | Claude, GPT-4o, Gemini, Groq | Paid / Free tier | N/A |

### ☁️ Cloud Providers

| Provider | Model String | Free Tier | API Key |
|----------|-------------|-----------|---------|
| Groq | `groq/llama-3.1-8b-instant` | ✅ Yes | [console.groq.com](https://console.groq.com) |
| OpenAI | `gpt-4o` | ❌ No | [platform.openai.com](https://platform.openai.com/api-keys) |
| Anthropic | `claude-sonnet-4-20250514` | ❌ No | [console.anthropic.com](https://console.anthropic.com) |
| Google | `gemini/gemini-pro` | ✅ Yes | [aistudio.google.com](https://aistudio.google.com/apikey) |

> 💡 **Groq** offers a generous free tier with fast inference. It's the best starting point if you want cloud models without paying.

Add API keys with: `shard model key <provider>`

---

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

### 🗄️ Redis Stack

Redis Stack provides the vector search engine for `shard ask`. It stores note embeddings and runs fast similarity queries.

Check if installed:

```bash
redis-cli ping
```

<details>
<summary>🐧 Arch Linux</summary>

```bash
yay -S redis-stack-server
sudo systemctl enable --now redis-stack-server
```

</details>

<details>
<summary>🐧 Ubuntu / Debian</summary>

```bash
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt update
sudo apt install redis-stack-server
sudo systemctl enable --now redis-stack-server
```

</details>

<details>
<summary>🍎 macOS</summary>

```bash
brew tap redis-stack/redis-stack
brew install redis-stack-server
redis-stack-server --daemonize yes
```

</details>

<details>
<summary>🪟 Windows</summary>

Use Docker:

```powershell
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest
```

</details>

<details>
<summary>🐳 Docker (any OS)</summary>

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest
```

</details>

Verify:

```bash
redis-cli ping  # Should print PONG
```

> 💡 Redis Stack must be running whenever you use `shard ask` or `shard index`. The `shard add` command indexes notes automatically.

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

---

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

## ⚙️ Configuration Reference

| Field | Default | Description |
|-------|---------|-------------|
| `vault_path` | *(set during setup)* | Absolute path to your Obsidian vault |
| `model` | `ollama_chat/qwen2.5:3b` | LiteLLM model string for note generation |
| `redis_host` | `localhost` | Redis Stack hostname |
| `redis_port` | `6379` | Redis Stack port |
| `redis_password` | `""` | Redis password (if set) |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |
| `custom_models` | `[]` | User-registered model descriptors |
| `api_keys` | `{}` | Provider API keys (alternative to env vars) |
| `notes_subfolder` | `""` (vault root) | Where new notes are saved |
| `style_profile` | auto-managed | Path to learned style JSON |

---

## 🏗️ How It Works

```text
Your Input → Extractor → Formatter (AI) → Indexer → Obsidian Vault
```

- **Extractor**: Pulls text from PDFs (pdfplumber), URLs (httpx + BeautifulSoup), YouTube (transcript API), or stdin
- **Formatter**: Sends text to your LLM to generate a structured note with title, tags, summary, and markdown body
- **Indexer**: Chunks the note text, generates vector embeddings (sentence-transformers), and stores them in Redis Stack for semantic search
- **Vault**: Saves the note as a markdown file with YAML frontmatter in your Obsidian vault

---

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
<summary>❌ Redis is not available</summary>

Redis Stack isn't running or doesn't have the RediSearch module loaded.

**Start Redis Stack:**

```bash
# Linux (systemd)
sudo systemctl start redis-stack-server

# macOS (Homebrew)
redis-stack-server --daemonize yes

# Docker
docker start redis-stack
```

If you have plain Redis without the Stack modules, you need Redis Stack instead. See the [Redis Stack prerequisite](#️-redis-stack).

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

---

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
