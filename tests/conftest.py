"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from shard.config import ShardConfig


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """Create a temporary Obsidian vault directory."""
    vault = tmp_path / "test-vault"
    vault.mkdir()
    return vault


@pytest.fixture
def mock_config(tmp_vault: Path, tmp_path: Path) -> ShardConfig:
    """Create a ShardConfig pointing at temporary directories."""
    return ShardConfig(
        vault_path=tmp_vault,
        chroma_path=tmp_path / ".chroma",
        model="ollama_chat/qwen2.5:3b",
    )
