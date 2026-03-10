"""Tests for shard.config — ShardConfig creation, load/save round-trip, and error paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shard.config import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    ShardConfig,
    _config_to_dict,
    _dict_to_config,
    load_config,
    save_config,
)
from shard.pipeline import ConfigError

# ── ShardConfig creation ──────────────────────────────────────────────────────


class TestShardConfigCreation:
    def test_minimal_creation_with_vault_path(self, tmp_vault: Path) -> None:
        cfg = ShardConfig(vault_path=tmp_vault)

        assert cfg.vault_path == tmp_vault
        assert cfg.model == DEFAULT_MODEL
        assert cfg.embedding_model == DEFAULT_EMBEDDING_MODEL
        assert cfg.chroma_path == DEFAULT_CHROMA_PATH
        assert cfg.custom_models == []
        assert cfg.api_keys == {}

    def test_vault_path_string_is_coerced_to_path(self, tmp_vault: Path) -> None:
        cfg = ShardConfig(vault_path=str(tmp_vault))

        assert isinstance(cfg.vault_path, Path)
        assert cfg.vault_path == tmp_vault

    def test_chroma_path_string_is_coerced_to_path(self, tmp_vault: Path, tmp_path: Path) -> None:
        chroma = tmp_path / "chroma"
        cfg = ShardConfig(vault_path=tmp_vault, chroma_path=str(chroma))

        assert isinstance(cfg.chroma_path, Path)
        assert cfg.chroma_path == chroma

    def test_full_construction(self, tmp_vault: Path, tmp_path: Path) -> None:
        chroma = tmp_path / ".chroma"
        cfg = ShardConfig(
            vault_path=tmp_vault,
            model="openai/gpt-4o",
            chroma_path=chroma,
            embedding_model="all-mpnet-base-v2",
            custom_models=[{"name": "gpt-4o", "provider": "openai"}],
            api_keys={"openai": "sk-test"},
        )

        assert cfg.model == "openai/gpt-4o"
        assert cfg.embedding_model == "all-mpnet-base-v2"
        assert cfg.custom_models == [{"name": "gpt-4o", "provider": "openai"}]
        assert cfg.api_keys == {"openai": "sk-test"}


# ── _dict_to_config ───────────────────────────────────────────────────────────


class TestDictToConfig:
    def test_raises_config_error_when_vault_path_missing(self) -> None:
        with pytest.raises(ConfigError, match="vault_path"):
            _dict_to_config({})

    def test_raises_config_error_with_only_unrelated_keys(self) -> None:
        with pytest.raises(ConfigError, match="vault_path"):
            _dict_to_config({"model": "ollama_chat/llama3"})

    def test_minimal_dict_uses_defaults(self, tmp_vault: Path) -> None:
        cfg = _dict_to_config({"vault_path": str(tmp_vault)})

        assert cfg.vault_path == tmp_vault
        assert cfg.model == DEFAULT_MODEL
        assert cfg.embedding_model == DEFAULT_EMBEDDING_MODEL
        assert cfg.custom_models == []
        assert cfg.api_keys == {}

    def test_all_fields_round_trip(self, tmp_vault: Path, tmp_path: Path) -> None:
        chroma = tmp_path / ".chroma"
        data = {
            "vault_path": str(tmp_vault),
            "model": "anthropic/claude-3-haiku",
            "chroma_path": str(chroma),
            "embedding_model": "paraphrase-MiniLM-L6-v2",
            "custom_models": [{"name": "haiku", "provider": "anthropic"}],
            "api_keys": {"anthropic": "sk-ant-test"},
        }

        cfg = _dict_to_config(data)

        assert cfg.vault_path == tmp_vault
        assert cfg.model == "anthropic/claude-3-haiku"
        assert cfg.chroma_path == chroma
        assert cfg.embedding_model == "paraphrase-MiniLM-L6-v2"
        assert cfg.custom_models == [{"name": "haiku", "provider": "anthropic"}]
        assert cfg.api_keys == {"anthropic": "sk-ant-test"}


# ── _config_to_dict ───────────────────────────────────────────────────────────


class TestConfigToDict:
    def test_paths_are_serialised_as_strings(self, mock_config: ShardConfig) -> None:
        d = _config_to_dict(mock_config)

        assert isinstance(d["vault_path"], str)
        assert isinstance(d["chroma_path"], str)

    def test_path_values_match_original(self, mock_config: ShardConfig) -> None:
        d = _config_to_dict(mock_config)

        assert d["vault_path"] == str(mock_config.vault_path)
        assert d["chroma_path"] == str(mock_config.chroma_path)


# ── save_config / load_config round-trip ─────────────────────────────────────


class TestSaveLoadRoundTrip:
    def test_saved_file_is_valid_json(self, tmp_path: Path, mock_config: ShardConfig) -> None:
        config_file = tmp_path / "config.json"
        save_config(mock_config, path=config_file)

        raw = config_file.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert "vault_path" in parsed

    def test_load_recovers_same_values(self, tmp_path: Path, mock_config: ShardConfig) -> None:
        config_file = tmp_path / "config.json"
        save_config(mock_config, path=config_file)
        loaded = load_config(path=config_file)

        assert loaded.vault_path == mock_config.vault_path
        assert loaded.model == mock_config.model
        assert loaded.chroma_path == mock_config.chroma_path
        assert loaded.embedding_model == mock_config.embedding_model

    def test_parent_dirs_are_created(self, tmp_path: Path, mock_config: ShardConfig) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "config.json"
        save_config(mock_config, path=deep_path)

        assert deep_path.exists()

    def test_round_trip_preserves_custom_models_and_api_keys(
        self, tmp_path: Path, tmp_vault: Path
    ) -> None:
        cfg = ShardConfig(
            vault_path=tmp_vault,
            custom_models=[{"name": "gpt-4o", "provider": "openai"}],
            api_keys={"openai": "sk-test-key"},
        )
        config_file = tmp_path / "cfg.json"
        save_config(cfg, path=config_file)
        loaded = load_config(path=config_file)

        assert loaded.custom_models == cfg.custom_models
        assert loaded.api_keys == cfg.api_keys


# ── load_config error paths ───────────────────────────────────────────────────


class TestLoadConfigErrors:
    def test_raises_when_file_does_not_exist(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"

        with pytest.raises(ConfigError, match="not found"):
            load_config(path=missing)

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ this is not valid json }", encoding="utf-8")

        with pytest.raises(ConfigError, match="invalid JSON"):
            load_config(path=bad_json)

    def test_raises_on_missing_vault_path_field(self, tmp_path: Path) -> None:
        config_file = tmp_path / "no_vault.json"
        config_file.write_text(json.dumps({"model": "ollama_chat/llama3"}), encoding="utf-8")

        with pytest.raises(ConfigError, match="vault_path"):
            load_config(path=config_file)
