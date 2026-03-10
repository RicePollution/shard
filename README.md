# Shard — AI-Powered Note Ingestion for Obsidian

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-success.svg)](https://ollama.com)

## What is Shard?

Shard is a CLI tool that ingests PDFs, URLs, YouTube videos, and text into your Obsidian vault as beautifully formatted, AI-generated notes with full semantic search. Everything runs locally and stays private—no accounts, no API keys required (unless you choose cloud LLMs).

<!-- Demo GIF: ![shard demo](assets/demo.gif) -->

## ✨ Features

- **📄 Multiple input formats** — Import from PDFs, web URLs, YouTube videos, or plain text
- **🤖 AI-powered formatting** — Automatically generates structured notes with summaries and tags
- **🔍 Semantic search** — Find notes by meaning, not just keywords
- **💾 Local-first** — Run Ollama locally for free, private note processing
- **🛡️ No vendor lock-in** — Export anytime; all notes live in your Obsidian vault
- **⚡ Fast & lightweight** — Runs on any modern machine, no GPU required
- **🔗 Obsidian integration** — Notes appear directly in your vault, searchable in Obsidian

## 📋 Prerequisites

Before installing Shard, ensure you have everything needed.

### 🐍 Python 3.11+

Python is the programming language Shard is built in. You need version 3.11 or higher.

**How to check:**
```bash
python --version
```
or
```bash
python3 --version
```

If you see `3.11.0` or higher, you're good. Otherwise, install it:

**Arch Linux:**
```bash
sudo pacman -S python
```
Verify:
```bash
python --version
```

**Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip
```
Verify:
```bash
python3 --version
```

**Fedora:**
```bash
sudo dnf install python3
```
Verify:
```bash
python3 --version
```

**macOS:**

Using Homebrew (recommended):
```bash
brew install python
```

Or download directly from [python.org](https://www.python.org/downloads/).

Verify:
```bash
python --version
```

**Windows:**

1. Download the installer from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important:** Check the box "Add Python to PATH"
4. Click Install

Verify in PowerShell or Command Prompt:
```powershell
python --version
```

### 📦 uv

`uv` is a fast, modern Python package manager. It's used to install Shard as a global CLI tool that you can run from anywhere without activating a virtual environment.

**What does "uv tool install" do?** It installs a Python app as a global command, like downloading an app on your phone. No virtual environments, no activation needed—just type `shard` anywhere.

**Install on Arch Linux:**
```bash
sudo pacman -S uv
```
Verify:
```bash
uv --version
```

**Install on Ubuntu / Debian:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```
Verify:
```bash
uv --version
```

**Install on Fedora:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```
Verify:
```bash
uv --version
```

**Install on macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
```
Verify:
```bash
uv --version
```

**Install on Windows:**

Open PowerShell as Administrator and run:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Then close and reopen PowerShell.

Verify:
```powershell
uv --version
```

### 🦙 Ollama

Ollama is free software that runs AI models locally on your machine. No internet connection needed after the initial download. No API keys, no accounts, completely private. Shard uses Ollama by default to keep everything on your device.

**Install on Arch Linux:**
```bash
sudo pacman -S ollama
```

**Install on Ubuntu / Debian:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Install on Fedora:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Install on macOS:**

Using Homebrew (recommended):
```bash
brew install ollama
```

Or download the installer directly from [ollama.com/download](https://ollama.com/download).

**Install on Windows:**

Download the installer from [ollama.com/download](https://ollama.com/download) and run it. Ollama will run as a system tray application.

**Start Ollama:**

**Linux (manual):**

Open a terminal and run:
```bash
ollama serve
```

Keep this terminal open while using Shard. Ollama listens for requests in the background.

**Linux (recommended — as a service):**

This starts Ollama automatically on boot so you never have to think about it:
```bash
sudo systemctl enable --now ollama
```

Verify it's running:
```bash
systemctl status ollama
```

**macOS:**

Ollama runs automatically as a menu bar app after you install it. No action needed.

**Windows:**

Ollama runs automatically as a tray app after you install it. No action needed.

> 💡 **Tip:** On Linux, using `systemctl enable --now ollama` is the easiest approach so Ollama starts automatically every time you boot.

**Pull the default model:**

Shard uses `qwen2.5:3b` by default. Download it now:
```bash
ollama pull qwen2.5:3b
```

This downloads about 2 GB of model weights. Depending on your internet speed, this may take 5-30 minutes. Only do this once.

> 💡 **Important:** This model runs on any modern machine. No GPU needed. It will be free to use forever. After download, it will be available offline whenever you need it.

**Verify Ollama works:**
```bash
ollama run qwen2.5:3b "say hello"
```

You should see a response from the model in your terminal.

### 🗃️ Obsidian Vault

You need Obsidian installed and at least one vault created. Shard saves notes directly into your vault's file system.

**Download Obsidian:** [obsidian.md](https://obsidian.md)

**Install on Arch Linux:**

Using AUR helper (e.g., `yay`):
```bash
yay -S obsidian
```

Or download the AppImage from [obsidian.md/download](https://obsidian.md).

**Install on Ubuntu / Debian:**

Download the `.deb` installer from [obsidian.md/download](https://obsidian.md/download) and run it:
```bash
sudo dpkg -i Obsidian-*.deb
```

**Install on Fedora:**

Download the `.rpm` installer from [obsidian.md/download](https://obsidian.md/download) and run it:
```bash
sudo dnf install ./Obsidian-*.rpm
```

**Install on macOS:**

Using Homebrew:
```bash
brew install --cask obsidian
```

Or download directly from [obsidian.md/download](https://obsidian.md/download).

**Install on Windows:**

Download the installer from [obsidian.md/download](https://obsidian.md/download) and run it.

**Create or locate your vault:**

You need the **folder path** to your vault. For example:
- Linux/macOS: `/home/yourname/Documents/MyVault` or `/Users/yourname/ObsidianVault`
- Windows: `C:\Users\yourname\Documents\MyVault`

If you don't have a vault yet, create a folder and open it in Obsidian:
1. Launch Obsidian
2. Click "Create new vault"
3. Choose a folder location (e.g., `~/Documents/MyVault`)
4. Give it a name (e.g., "MyVault")
5. Click Create

You'll need the vault folder path in the next section.

## 🚀 Installation

Now install Shard itself.

**Clone the repository:**
```bash
git clone https://github.com/RicePollution/shard
cd shard
```

**Install as a global command:**
```bash
uv tool install .
```

This installs Shard as a command-line tool. You only need to do this once.

**Verify installation:**
```bash
shard --help
```

You should see the Shard help menu.

> 💡 **If `shard: command not found` appears**, the Shard binary is not in your PATH. Fix this:

**Linux / macOS:**

Add this line to `~/.bashrc` (or `~/.zshrc` if you use Zsh):
```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then reload your shell:
```bash
source ~/.bashrc
```

**Windows:**

1. Press `Win + X` and click "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables," click "New"
5. Variable name: `PATH`
6. Variable value: `%APPDATA%\Python\Scripts`
7. Click OK three times
8. Restart PowerShell

Then verify:
```powershell
shard --help
```

## ⚙️ First Time Setup

On your first run, Shard will ask a few questions to set up your configuration. This is a one-time wizard.

**Run the wizard:**
```bash
shard config
```

You'll be prompted with these questions:

**1. Vault path:**
```
Path to your Obsidian vault [/home/yourname/Documents/ObsidianVault]:
```

Enter the folder path to your Obsidian vault. Press Enter to accept the default or type your custom path.

**2. Ollama check:**

Shard will check if Ollama is running:
```
Checking for a local Ollama installation…
Ollama is running. Found 1 installed model(s).
```

If you see "Could not reach Ollama," ensure `ollama serve` is running (or the service is started).

**3. Model selection:**

If Shard detects your Ollama installation:
```
Installed models:
  ollama_chat/qwen2.5:3b

'qwen2.5:3b' is already installed. Using it as the default model.
```

Or if the model isn't installed:
```
'qwen2.5:3b' is not installed.
Pull 'qwen2.5:3b' now? (~2 GB download) [Y/n]:
```

Type `y` to download, or `n` if you've already downloaded it manually.

**4. Config saved:**

```
Config saved to /home/yourname/.config/shard/config.json
```

Your configuration is now saved. You won't see this wizard again unless you run `shard config --setup`.

**Config file locations:**
- Linux/macOS: `~/.config/shard/config.json`
- Windows: `%APPDATA%\shard\config.json`

You can view or edit this file directly, but usually you'll manage it with `shard config`.

## 📖 Usage

Now you're ready to use Shard. Here are all 6 commands.

### shard add

Add a note from a file, URL, YouTube video, or text.

**Syntax:**
```bash
shard add INPUT
```

`INPUT` can be:
- A local file path (PDF or text)
- A web URL
- A YouTube video URL
- Any freeform text (enclosed in quotes if it contains spaces)

**Examples:**

**Example 1: Add a PDF file:**
```bash
shard add ~/Downloads/research_paper.pdf
```

Output:
```
Extracted: pdf — research_paper
Formatted: Research Paper on Neural Networks (8 tags)
Saved: Imported/Shards/Research_Paper_on_Neural_Networks.md
Indexed: 12 chunks

Shard added successfully.
  Title:  Research Paper on Neural Networks
  Tags:   machine-learning, deep-learning, transformers, nlp, research, pytorch, attention, optimization
  Chunks: 12
  Path:   Imported/Shards/Research_Paper_on_Neural_Networks.md
```

**Example 2: Add a webpage:**
```bash
shard add https://en.wikipedia.org/wiki/Machine_learning
```

Output:
```
Extracted: url — Machine learning - Wikipedia
Formatted: Machine Learning Fundamentals (5 tags)
Saved: Imported/Shards/Machine_Learning_Fundamentals.md
Indexed: 8 chunks

Shard added successfully.
  Title:  Machine Learning Fundamentals
  Tags:   algorithms, data-science, supervised-learning, unsupervised-learning, ai
  Chunks: 8
  Path:   Imported/Shards/Machine_Learning_Fundamentals.md
```

**Example 3: Add a YouTube video:**
```bash
shard add https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

Output:
```
Extracted: youtube — Rick Astley - Never Gonna Give You Up
Formatted: Never Gonna Give You Up Analysis (3 tags)
Saved: Imported/Shards/Never_Gonna_Give_You_Up_Analysis.md
Indexed: 4 chunks

Shard added successfully.
  Title:  Never Gonna Give You Up Analysis
  Tags:   music, 1987, pop
  Chunks: 4
  Path:   Imported/Shards/Never_Gonna_Give_You_Up_Analysis.md
```

**Example 4: Add plain text directly:**
```bash
shard add "Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications across clusters of machines."
```

Output:
```
Extracted: text — Untitled
Formatted: Kubernetes Container Orchestration (4 tags)
Saved: Imported/Shards/Kubernetes_Container_Orchestration.md
Indexed: 2 chunks

Shard added successfully.
  Title:  Kubernetes Container Orchestration
  Tags:   containers, devops, orchestration, infrastructure
  Chunks: 2
  Path:   Imported/Shards/Kubernetes_Container_Orchestration.md
```

**Example 5: Add from stdin (pipe text in):**
```bash
cat my_notes.txt | shard add -
```

Or:
```bash
echo "This is a quick note I want to save" | shard add -
```

### shard ask

Query your notes with a natural language question. Shard finds the most relevant notes and uses AI to generate an answer.

**Syntax:**
```bash
shard ask QUESTION [--top-k N]
```

- `QUESTION` — What you want to know (in quotes if it has spaces)
- `--top-k N` — Number of note chunks to use as context (default: 5)

**Example 1: Basic question:**
```bash
shard ask "What is machine learning?"
```

Output:
```
Searching notes…

Machine learning is a subset of artificial intelligence that enables systems to
learn and improve from experience without being explicitly programmed. Based on
your notes, machine learning algorithms can be categorized into supervised
learning, unsupervised learning, and reinforcement learning. Supervised learning
uses labeled data to train models, while unsupervised learning finds patterns in
unlabeled data. Reinforcement learning involves training agents to make sequential
decisions through rewards and penalties.

Sources
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Title                     ┃ Path                   ┃ Relevance ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ ML Fundamentals           │ Imported/Shards/ML... │ 94.2%     │
│ AI Overview               │ Imported/Shards/AI... │ 87.5%     │
└───────────────────────────┴────────────────────────┴───────────┘
```

**Example 2: Question with custom relevance:**
```bash
shard ask "How does reinforcement learning work?" --top-k 10
```

Output:
```
Searching notes…

Reinforcement learning is a machine learning technique where an agent learns by
interacting with an environment. The agent receives feedback in the form of
rewards or penalties (called the reward signal) for its actions. Over time, it
learns a policy—a strategy for choosing actions—that maximizes cumulative reward.
This approach is used in game playing (like AlphaGo), robotics, and autonomous
systems.

Sources
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Title                     ┃ Path                   ┃ Relevance ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Advanced ML Techniques    │ Imported/Shards/Ad... │ 96.3%     │
│ Learning Systems          │ Imported/Shards/Le... │ 91.2%     │
│ Game AI                   │ Imported/Shards/Ga... │ 85.7%     │
└───────────────────────────┴────────────────────────┴───────────┘
```

### shard index

Rebuild the search index by scanning all notes in your vault and adding them to the search database (ChromaDB).

Run this if you've manually edited notes in Obsidian or added notes outside of Shard, and you want them searchable.

**Syntax:**
```bash
shard index
```

**Example:**
```bash
shard index
```

Output:
```
Reindex complete. Total chunks: 147
```

This scanned all notes in `<vault>/Imported/Shards/`, split them into chunks, and added them to the search index.

### shard list

Show a table of all imported shard notes in your vault.

**Syntax:**
```bash
shard list
```

**Example:**
```bash
shard list
```

Output:
```
Shard Notes (23)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Title                     ┃ Tags                ┃ Date       ┃ Path                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Machine Learning Basics   │ ai, learning        │ 2024-01-15 │ Imported/Shards/ML... │
│ Python Best Practices     │ python, code        │ 2024-01-14 │ Imported/Shards/Py... │
│ Web Architecture          │ web, systems        │ 2024-01-13 │ Imported/Shards/We... │
│ Kubernetes Guide          │ devops, containers  │ 2024-01-12 │ Imported/Shards/Ku... │
│ React Hooks Deep Dive     │ react, javascript   │ 2024-01-11 │ Imported/Shards/Re... │
└───────────────────────────┴─────────────────────┴────────────┴────────────────────────┘
```

### shard open

Open a shard note in Obsidian by fuzzy-matching its title.

**Syntax:**
```bash
shard open QUERY
```

`QUERY` is a partial title. Shard finds the best match using fuzzy matching.

**Example 1:**
```bash
shard open "machine"
```

Output:
```
Matched: Machine Learning Basics (score 95/100)
Opened in Obsidian.
```

Obsidian will open and show the matched note.

**Example 2:**
```bash
shard open "react"
```

Output:
```
Matched: React Hooks Deep Dive (score 87/100)
Opened in Obsidian.
```

### shard config

View or update your Shard configuration.

**Syntax (view all settings):**
```bash
shard config
```

**Example output:**
```
Shard Configuration  /home/yourname/.config/shard/config.json
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Key            ┃ Value                  ┃ Description             ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ vault_path     │ /home/yours/MyVault    │ Absolute path to vault  │
│ model          │ ollama_chat/qwen2.5:3b │ LLM for formatting      │
│ chroma_path    │ ~/.local/share/shard.. │ Search database path    │
│ embedding_..   │ all-MiniLM-L6-v2       │ Embedding model         │
└────────────────┴────────────────────────┴─────────────────────────┘
```

**Syntax (update a single setting):**
```bash
shard config --set KEY=VALUE
```

**Example 1: Change the model:**
```bash
shard config --set model=ollama_chat/llama3.1:8b
```

Output:
```
Config updated: model = ollama_chat/llama3.1:8b
Saved to /home/yourname/.config/shard/config.json
```

**Example 2: Change the vault path:**
```bash
shard config --set vault_path=/home/yourname/Documents/NewVault
```

Output:
```
Config updated: vault_path = /home/yourname/Documents/NewVault
Saved to /home/yourname/.config/shard/config.json
```

**Syntax (rerun the setup wizard):**
```bash
shard config --setup
```

This walks you through the initial setup again, useful if you want to change your vault or model without editing JSON.

## 🤖 Models

Shard supports multiple AI models. The default is `qwen2.5:3b` running locally via Ollama, but you can switch to larger local models or cloud providers.

### Local Models (Free, Private)

These run entirely on your machine using Ollama.

| Tier | Model Names | Cost | RAM | Speed |
|------|-------------|------|-----|-------|
| 🟢 Small | `qwen2.5:3b`, `phi3.5`, `llama3.2:3b` | Free | 4 GB | Fast |
| 🟡 Large | `llama3.1:8b`, `qwen2.5:14b` | Free | 8+ GB | Slower |

**Switch to a different local model:**

First, pull it from Ollama:
```bash
ollama pull llama3.1:8b
```

Then configure Shard to use it:
```bash
shard config --set model=ollama_chat/llama3.1:8b
```

Verify:
```bash
shard config
```

### Cloud Models (Paid, No Setup)

For higher quality or to offload computation, use a cloud LLM provider.

| Provider | Model String | Free Tier | Get API Key |
|----------|--------------|-----------|-------------|
| OpenAI (GPT-4o) | `gpt-4o` | $5 credit | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Anthropic (Claude) | `claude-opus-4-1-20250805` | $5 credit | [console.anthropic.com/keys](https://console.anthropic.com/keys) |
| Google (Gemini) | `gemini-2.0-flash` | Free tier | [ai.google.dev/pricing](https://ai.google.dev/pricing) |
| Groq (Free & Fast) | `groq/mixtral-8x7b-32768` | Free, no card | [console.groq.com](https://console.groq.com) |

> 💡 **Recommendation:** Groq offers the best value—it's completely free, no credit card required, and incredibly fast. Perfect for trying cloud models risk-free.

**Set up a cloud model:**

**1. Get your API key** from the provider's website (see table above).

**2. Set the API key as an environment variable:**

**Linux / macOS:**
```bash
export OPENAI_API_KEY="sk-..."
```

Add this to `~/.bashrc` or `~/.zshrc` to make it permanent:
```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

To make it permanent, use System Properties → Environment Variables (see Python installation section).

**3. Configure Shard to use the model:**
```bash
shard config --set model=gpt-4o
```

**4. Verify:**
```bash
shard config
```

**Example: Use Claude from Anthropic:**

1. Get your API key from [console.anthropic.com/keys](https://console.anthropic.com/keys)
2. Set the environment variable:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```
3. Configure Shard:
   ```bash
   shard config --set model=claude-opus-4-1-20250805
   ```

**Example: Use Groq (free, no card required):**

1. Sign up at [console.groq.com](https://console.groq.com) (no credit card needed)
2. Get your API key from the dashboard
3. Set the environment variable:
   ```bash
   export GROQ_API_KEY="gsk_..."
   ```
4. Configure Shard:
   ```bash
   shard config --set model=groq/mixtral-8x7b-32768
   ```

> 💡 **Note:** Any model supported by [litellm](https://litellm.ai) can be used. Check their documentation for the full list.

## ⚙️ Configuration Reference

All settings stored in `~/.config/shard/config.json` (or `%APPDATA%\shard\config.json` on Windows).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vault_path` | Path | *(required)* | Absolute path to your Obsidian vault directory. Shard saves notes to `<vault>/Imported/Shards/`. |
| `model` | String | `ollama_chat/qwen2.5:3b` | LiteLLM model string for generating note titles, tags, and summaries. Examples: `gpt-4o`, `claude-opus-4-1-20250805`, `groq/mixtral-8x7b-32768`, `ollama_chat/llama3.1:8b`. |
| `chroma_path` | Path | `~/.local/share/shard/chroma` | Directory where ChromaDB persists the vector search index. Shard creates this automatically. |
| `embedding_model` | String | `all-MiniLM-L6-v2` | Sentence-transformers embedding model name for semantic search. Must be a valid Hugging Face model ID. Rarely needs to change. |
| `custom_models` | List | `[]` | Advanced: list of custom model definitions. Used for registering non-standard LLM endpoints. |
| `api_keys` | Dict | `{}` | Advanced: mapping of provider names to API keys (e.g. `{"openai": "sk-...", "anthropic": "sk-ant-..."}`). Better to use environment variables. |

## 🏗️ How It Works

Shard processes content through a three-stage pipeline:

```
Your Input (PDF, URL, YouTube, text)
  ↓
[Extractor] — Pulls text from the source
  ↓
[Formatter] — Uses AI to generate title, tags, and summary
  ↓
[Indexer] — Splits note into chunks and adds to search database
  ↓
Obsidian Vault — Note appears in <vault>/Imported/Shards/
```

**Stage 1: Extraction**

Shard detects the input type (PDF, web URL, YouTube, or plain text) and extracts the raw text content. PDFs are parsed with `pdfplumber`. Web pages are downloaded and cleaned with `BeautifulSoup`. YouTube transcripts are fetched via the YouTube Transcript API. Plain text is used as-is.

**Stage 2: Formatting**

The extracted text is sent to your configured LLM (default: Ollama's `qwen2.5:3b`) with a prompt asking it to:
- Generate a meaningful title
- Extract 3-8 relevant tags
- Write a concise summary
- Preserve the full body text

The result is a structured note with YAML frontmatter (metadata).

**Stage 3: Indexing**

The formatted note is saved to your Obsidian vault. Then:
- The note body is split into semantic chunks (paragraphs)
- Each chunk is embedded using `sentence-transformers`
- Embeddings are stored in ChromaDB for semantic search

Now the note is fully searchable with `shard ask`.

## 🔧 Troubleshooting

**"shard: command not found"**

The `shard` binary is not in your system PATH. See the Installation section above under "If `shard: command not found` appears" for your OS.

**"Connection refused" when running shard add or shard ask**

Ollama is not running. Make sure to:
- **Linux (manual):** Run `ollama serve` in a separate terminal
- **Linux (service):** Verify with `systemctl status ollama`
- **macOS/Windows:** Ensure Ollama is launched (it runs in background)

Then retry your command.

**"Model not found" error**

The model you configured is not installed locally. Pull it:
```bash
ollama pull qwen2.5:3b
```

Or switch to a model you've already pulled:
```bash
shard config --set model=ollama_chat/llama3.2:3b
```

**"Vault path not found"**

The vault path in your config no longer exists or is wrong. Fix it:
```bash
shard config
```

This reruns the setup wizard. Or manually update:
```bash
shard config --set vault_path=/path/to/your/vault
```

**"Slow on first run"**

The first time you run `shard add` or `shard ask`, Shard must load the embedding model into memory. This takes 30-60 seconds on first load. Subsequent runs are much faster. This is normal.

**Search returns no results or irrelevant results**

Make sure you've indexed your notes:
```bash
shard index
```

This rebuilds the search database. Also, `shard ask` uses semantic search, so phrasing matters. Try rephrasing your question in different ways.

**Notes not appearing in Obsidian**

Check that the vault path is correct:
```bash
shard config
```

And verify that notes are being saved to the right location. They should appear in `<vault>/Imported/Shards/`.

If you moved your vault, update the config:
```bash
shard config --set vault_path=/new/vault/path
```

## 🤝 Contributing

Shard is open source! We welcome contributions.

**Good first issues:**
- Add support for new input formats (EPUB, HTML files, etc.)
- Improve embedding model selection UI
- Add more model provider integrations
- Expand documentation and tutorials
- Performance optimizations for large vaults

**How to contribute:**
1. Fork the repository on GitHub
2. Create a branch for your feature
3. Make changes and test locally
4. Submit a pull request with a clear description

## 📄 License

Shard is licensed under the MIT License. See the LICENSE file for details.

Free to use, modify, and distribute for any purpose.
