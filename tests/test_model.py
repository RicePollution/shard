"""Tests for the `shard model` CLI command group and supporting model utilities.

Covers:
- shard model             (status display, no subcommand)
- shard model list        (tier display, current-model marker, pulled/key status)
- shard model use <model> (config switch, Ollama pull prompt, cloud key prompt)
- shard model key <prov>  (save, mask, remove, empty-input, bad-prefix warnings)
- complete()              (AuthenticationError → friendly ModelError)
- inject_api_keys()       (env-var injection, no-override semantics)
- _detect_provider()      (model-string → provider routing)

All tests mock external I/O (Ollama API, litellm, config I/O) so no network
access or local services are required.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import litellm
import pytest
from click.testing import CliRunner

from shard.cli import cli
from shard.config import ShardConfig
from shard.pipeline import ModelError

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_config(tmp_path: Path, *, model: str = "ollama_chat/qwen2.5:3b") -> ShardConfig:
    """Return a minimal ShardConfig pointing at temporary directories."""
    vault = tmp_path / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    return ShardConfig(
        vault_path=vault,
        chroma_path=tmp_path / ".chroma",
        model=model,
    )


def _fake_config_path(tmp_path: Path) -> Path:
    """Write a placeholder config file and return its path."""
    p = tmp_path / "config.json"
    p.write_text("{}", encoding="utf-8")
    return p


# Minimal list_models() return value used across several test classes.
_SAMPLE_MODELS: list[dict] = [
    {
        "name": "ollama_chat/qwen2.5:3b",
        "label": "Qwen 2.5 3B",
        "tier": "local_small",
        "provider": "ollama",
        "current": True,
        "pulled": True,
        "has_key": None,
        "free": False,
    },
    {
        "name": "ollama_chat/llama3.1:8b",
        "label": "Llama 3.1 8B",
        "tier": "local_large",
        "provider": "ollama",
        "current": False,
        "pulled": False,
        "has_key": None,
        "free": False,
    },
    {
        "name": "gpt-4o",
        "label": "GPT-4o",
        "tier": "cloud",
        "provider": "openai",
        "current": False,
        "pulled": None,
        "has_key": True,
        "free": False,
    },
    {
        "name": "groq/llama3-70b",
        "label": "Llama 3 70B (Groq)",
        "tier": "cloud",
        "provider": "groq",
        "current": False,
        "pulled": None,
        "has_key": False,
        "free": True,
    },
]


# ── TestModelStatus ───────────────────────────────────────────────────────────


class TestModelStatus:
    """shard model — invoked with no subcommand."""

    def test_model_shows_current_model(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path, model="ollama_chat/qwen2.5:3b")
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
        ):
            result = runner.invoke(cli, ["model"])

        assert result.exit_code == 0
        assert "ollama_chat/qwen2.5:3b" in result.output

    def test_model_shows_subcommand_hints(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
        ):
            result = runner.invoke(cli, ["model"])

        assert result.exit_code == 0
        # The status view must hint at the key sub-commands.
        for hint in ("model use", "model pull", "model key", "model list"):
            assert hint in result.output

    def test_model_shows_not_set_when_model_empty(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path, model="")
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
        ):
            result = runner.invoke(cli, ["model"])

        assert result.exit_code == 0
        # Rich strips markup, but the plain-text fallback should surface.
        assert "not set" in result.output

    def test_model_propagates_config_error(self, tmp_path: Path) -> None:
        from shard.pipeline import ConfigError

        runner = CliRunner()
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", side_effect=ConfigError("bad config")),
        ):
            result = runner.invoke(cli, ["model"])

        assert result.exit_code == 1
        assert "bad config" in result.output


# ── TestModelList ─────────────────────────────────────────────────────────────


class TestModelList:
    """shard model list — tier display, current marker, pulled/key status."""

    def test_list_shows_local_small_tier(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.models.list_models", return_value=_SAMPLE_MODELS),
        ):
            result = runner.invoke(cli, ["model", "list"])

        assert result.exit_code == 0
        assert "Local Small" in result.output

    def test_list_shows_local_large_tier(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.models.list_models", return_value=_SAMPLE_MODELS),
        ):
            result = runner.invoke(cli, ["model", "list"])

        assert result.exit_code == 0
        assert "Local Large" in result.output

    def test_list_shows_cloud_tier(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.models.list_models", return_value=_SAMPLE_MODELS),
        ):
            result = runner.invoke(cli, ["model", "list"])

        assert result.exit_code == 0
        assert "Cloud" in result.output

    def test_list_marks_current_model(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path, model="ollama_chat/qwen2.5:3b")
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.models.list_models", return_value=_SAMPLE_MODELS),
        ):
            result = runner.invoke(cli, ["model", "list"])

        assert result.exit_code == 0
        assert "current" in result.output

    def test_list_shows_pulled_indicator_for_ollama(self, tmp_path: Path) -> None:
        """Pulled Ollama models show a check-mark; un-pulled ones show 'not pulled'."""
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.models.list_models", return_value=_SAMPLE_MODELS),
        ):
            result = runner.invoke(cli, ["model", "list"])

        assert result.exit_code == 0
        assert "not pulled" in result.output

    def test_list_shows_key_set_for_cloud_model_with_key(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.models.list_models", return_value=_SAMPLE_MODELS),
        ):
            result = runner.invoke(cli, ["model", "list"])

        assert result.exit_code == 0
        assert "key set" in result.output

    def test_list_shows_no_key_for_cloud_model_without_key(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.models.list_models", return_value=_SAMPLE_MODELS),
        ):
            result = runner.invoke(cli, ["model", "list"])

        assert result.exit_code == 0
        assert "no key" in result.output

    def test_list_shows_model_names(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.models.list_models", return_value=_SAMPLE_MODELS),
        ):
            result = runner.invoke(cli, ["model", "list"])

        assert result.exit_code == 0
        assert "qwen2.5:3b" in result.output
        assert "gpt-4o" in result.output

    def test_list_propagates_shard_error(self, tmp_path: Path) -> None:
        from shard.pipeline import ModelError

        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.models.list_models", side_effect=ModelError("Ollama down")),
        ):
            result = runner.invoke(cli, ["model", "list"])

        assert result.exit_code == 1
        assert "Ollama down" in result.output

    def test_list_skips_empty_tiers(self, tmp_path: Path) -> None:
        """When a tier has no models it must not appear in output."""
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        # Only cloud models in the sample — Local Small / Large must be absent.
        cloud_only = [m for m in _SAMPLE_MODELS if m["tier"] == "cloud"]

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.models.list_models", return_value=cloud_only),
        ):
            result = runner.invoke(cli, ["model", "list"])

        assert result.exit_code == 0
        assert "Local Small" not in result.output
        assert "Local Large" not in result.output
        assert "Cloud" in result.output


# ── TestModelUse ──────────────────────────────────────────────────────────────


class TestModelUse:
    """shard model use <model> — config switch, pull prompt, key prompt."""

    def test_use_switches_config_for_pulled_ollama(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path, model="ollama_chat/qwen2.5:3b")
        cfg_path = _fake_config_path(tmp_path)
        mock_save = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config", mock_save),
            patch(
                "shard.models.detect_available_models",
                return_value=["ollama_chat/llama3.2:3b"],
            ),
        ):
            result = runner.invoke(cli, ["model", "use", "ollama_chat/llama3.2:3b"])

        assert result.exit_code == 0
        mock_save.assert_called_once()
        saved_config: ShardConfig = mock_save.call_args[0][0]
        assert saved_config.model == "ollama_chat/llama3.2:3b"

    def test_use_confirms_success_message(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config"),
            patch(
                "shard.models.detect_available_models",
                return_value=["ollama_chat/llama3.2:3b"],
            ),
        ):
            result = runner.invoke(cli, ["model", "use", "ollama_chat/llama3.2:3b"])

        assert "llama3.2:3b" in result.output

    def test_use_prompts_pull_for_missing_ollama_and_user_accepts(
        self, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)
        mock_save = MagicMock()
        mock_pull = MagicMock(return_value=True)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config", mock_save),
            patch("shard.models.detect_available_models", return_value=[]),
            patch("shard.models.pull_ollama_model", mock_pull),
        ):
            # Simulate user pressing Enter (accept default "y") at the pull prompt.
            result = runner.invoke(
                cli, ["model", "use", "ollama_chat/phi3.5"], input="y\n"
            )

        assert result.exit_code == 0
        mock_pull.assert_called_once_with("phi3.5")
        mock_save.assert_called_once()

    def test_use_aborts_when_user_declines_pull(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)
        mock_save = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config", mock_save),
            patch("shard.models.detect_available_models", return_value=[]),
            patch("shard.models.pull_ollama_model", return_value=True),
        ):
            result = runner.invoke(
                cli, ["model", "use", "ollama_chat/phi3.5"], input="n\n"
            )

        # User declined: exit 0, save_config never called.
        assert result.exit_code == 0
        mock_save.assert_not_called()

    def test_use_exits_nonzero_when_pull_fails(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config"),
            patch("shard.models.detect_available_models", return_value=[]),
            patch("shard.models.pull_ollama_model", return_value=False),
        ):
            result = runner.invoke(
                cli, ["model", "use", "ollama_chat/phi3.5"], input="y\n"
            )

        assert result.exit_code == 1
        assert "Pull failed" in result.output or "failed" in result.output.lower()

    def test_use_prompts_key_for_cloud_model_without_key(self, tmp_path: Path) -> None:
        """Selecting a cloud model with no key should ask if the user wants to add one."""
        runner = CliRunner()
        # No API key set in config, no env var.
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)
        mock_save = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config", mock_save),
            patch.dict(os.environ, {}, clear=False),
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False),
        ):
            # Accept key prompt, then provide a key.
            result = runner.invoke(
                cli,
                ["model", "use", "gpt-4o"],
                input="y\nsk-test-key-abcdefghijk\n",
            )

        assert result.exit_code == 0
        mock_save.assert_called_once()
        saved: ShardConfig = mock_save.call_args[0][0]
        assert saved.model == "gpt-4o"
        assert saved.api_keys.get("openai") == "sk-test-key-abcdefghijk"

    def test_use_aborts_when_user_declines_key_prompt(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)
        mock_save = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config", mock_save),
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False),
        ):
            result = runner.invoke(cli, ["model", "use", "gpt-4o"], input="n\n")

        # Declined: exit 0 (graceful abort), model not saved.
        assert result.exit_code == 0
        mock_save.assert_not_called()

    def test_use_skips_key_prompt_when_env_var_is_set(self, tmp_path: Path) -> None:
        """If OPENAI_API_KEY is already in the environment, no key prompt should appear."""
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)
        mock_save = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config", mock_save),
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-already-set"}, clear=False),
        ):
            result = runner.invoke(cli, ["model", "use", "gpt-4o"])

        assert result.exit_code == 0
        mock_save.assert_called_once()
        saved: ShardConfig = mock_save.call_args[0][0]
        assert saved.model == "gpt-4o"

    def test_use_skips_key_prompt_when_key_in_config(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        config.api_keys["openai"] = "sk-from-config"
        cfg_path = _fake_config_path(tmp_path)
        mock_save = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config", mock_save),
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False),
        ):
            result = runner.invoke(cli, ["model", "use", "gpt-4o"])

        assert result.exit_code == 0
        mock_save.assert_called_once()


# ── TestModelKey ──────────────────────────────────────────────────────────────


class TestModelKey:
    """shard model key — add/update, list (masked), remove, validation."""

    def test_key_saves_to_config(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)
        mock_save = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config", mock_save),
        ):
            result = runner.invoke(
                cli, ["model", "key", "openai"], input="sk-test12345abcde\n"
            )

        assert result.exit_code == 0
        mock_save.assert_called_once()
        saved: ShardConfig = mock_save.call_args[0][0]
        assert saved.api_keys["openai"] == "sk-test12345abcde"

    def test_key_sets_env_var_on_save(self, tmp_path: Path) -> None:
        """After a key is accepted, the matching env var must be set before returning."""
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        # Use an isolated environment so the real OPENAI_API_KEY doesn't interfere.
        isolated_env: dict[str, str] = {}

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config"),
            patch.dict(os.environ, isolated_env, clear=True),
        ):
            runner.invoke(cli, ["model", "key", "openai"], input="sk-testXYZ98765\n")
            # Read the env var inside the patch context while it is still active.
            key_in_env = os.environ.get("OPENAI_API_KEY", "")

        assert key_in_env == "sk-testXYZ98765"

    def test_key_list_shows_masked_keys(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        config.api_keys["openai"] = "sk-verylongkeyvalue12345"
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
        ):
            result = runner.invoke(cli, ["model", "key", "--list"])

        assert result.exit_code == 0
        # Full key must NOT appear.
        assert "sk-verylongkeyvalue12345" not in result.output
        # Masked form: first 6 chars + "..." + last 5 chars.
        assert "sk-ver" in result.output
        assert "12345" in result.output

    def test_key_list_shows_not_set_for_missing_keys(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        # No api_keys set.
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
        ):
            result = runner.invoke(cli, ["model", "key", "--list"])

        assert result.exit_code == 0
        assert "not set" in result.output

    def test_key_remove_deletes_key_from_config(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        config.api_keys["anthropic"] = "sk-ant-existing"
        cfg_path = _fake_config_path(tmp_path)
        mock_save = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config", mock_save),
        ):
            result = runner.invoke(cli, ["model", "key", "--remove", "anthropic"])

        assert result.exit_code == 0
        mock_save.assert_called_once()
        saved: ShardConfig = mock_save.call_args[0][0]
        assert "anthropic" not in saved.api_keys

    def test_key_remove_warns_when_key_not_set(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)  # No anthropic key.
        cfg_path = _fake_config_path(tmp_path)
        mock_save = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config", mock_save),
        ):
            result = runner.invoke(cli, ["model", "key", "--remove", "anthropic"])

        assert result.exit_code == 0
        # save_config must NOT be called — nothing to remove.
        mock_save.assert_not_called()
        assert "No key configured" in result.output

    def test_key_rejects_empty_input(self, tmp_path: Path) -> None:
        """Providing only whitespace/newline at the key prompt must not save."""
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)
        mock_save = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config", mock_save),
        ):
            # click.prompt(hide_input=True) aborts when the user sends only a
            # newline — CliRunner surfaces that as exit code 1 with "Aborted!".
            result = runner.invoke(cli, ["model", "key", "openai"], input="\n")

        assert result.exit_code == 1
        mock_save.assert_not_called()

    def test_key_warns_on_bad_prefix_but_still_saves(self, tmp_path: Path) -> None:
        """A key with an unexpected prefix triggers a warning but is accepted."""
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)
        mock_save = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config", mock_save),
        ):
            # Valid-length key but wrong prefix (expected "sk-" for openai).
            result = runner.invoke(
                cli, ["model", "key", "openai"], input="bad-prefix-key-value\n"
            )

        assert result.exit_code == 0
        assert "Warning" in result.output or "warning" in result.output.lower()
        mock_save.assert_called_once()
        saved: ShardConfig = mock_save.call_args[0][0]
        assert saved.api_keys["openai"] == "bad-prefix-key-value"

    def test_key_rejects_unknown_provider(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
        ):
            result = runner.invoke(cli, ["model", "key", "notarealprovider"])

        assert result.exit_code == 1
        assert "Unknown provider" in result.output or "unknown" in result.output.lower()

    def test_key_requires_provider_argument(self, tmp_path: Path) -> None:
        """Invoking `shard model key` with no provider should print an error."""
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
        ):
            result = runner.invoke(cli, ["model", "key"])

        assert result.exit_code == 1
        assert "provider" in result.output.lower() or "Error" in result.output

    def test_key_validates_anthropic_prefix(self, tmp_path: Path) -> None:
        """Providing correct sk-ant- prefix produces no warning."""
        runner = CliRunner()
        config = _make_config(tmp_path)
        cfg_path = _fake_config_path(tmp_path)

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config"),
        ):
            result = runner.invoke(
                cli,
                ["model", "key", "anthropic"],
                input="sk-ant-validkeyvalue12345\n",
            )

        assert result.exit_code == 0
        assert "Warning" not in result.output

    def test_key_remove_clears_env_var(self, tmp_path: Path) -> None:
        """Removing a key should also delete its environment variable."""
        runner = CliRunner()
        config = _make_config(tmp_path)
        config.api_keys["openai"] = "sk-to-be-removed"
        cfg_path = _fake_config_path(tmp_path)

        # Use an isolated env dict so the real OPENAI_API_KEY never bleeds in.
        isolated_env = {"OPENAI_API_KEY": "sk-to-be-removed"}

        with (
            patch("shard.cli.CONFIG_PATH", cfg_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config"),
            patch.dict(os.environ, isolated_env, clear=True),
        ):
            runner.invoke(cli, ["model", "key", "--remove", "openai"])
            key_after = os.environ.get("OPENAI_API_KEY", "REMOVED")

        assert key_after == "REMOVED"


# ── TestModelPull ─────────────────────────────────────────────────────────────


class TestModelPull:
    """Tests for ``shard model pull <model>``."""

    def test_pull_strips_prefix_and_calls_pull(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config"),
            patch("shard.models.pull_ollama_model", return_value=True) as mock_pull,
        ):
            result = runner.invoke(cli, ["model", "pull", "ollama_chat/llama3.1:8b"], input="y\n")

        mock_pull.assert_called_once_with("llama3.1:8b")
        assert result.exit_code == 0

    def test_pull_bare_name_works(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config"),
            patch("shard.models.pull_ollama_model", return_value=True) as mock_pull,
        ):
            result = runner.invoke(cli, ["model", "pull", "phi3.5"], input="y\n")

        mock_pull.assert_called_once_with("phi3.5")
        assert result.exit_code == 0

    def test_pull_failure_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.models.pull_ollama_model", return_value=False),
        ):
            result = runner.invoke(cli, ["model", "pull", "bad-model"])

        assert result.exit_code == 1
        assert "Failed" in result.output

    def test_pull_success_prompts_set_default(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config") as mock_save,
            patch("shard.models.pull_ollama_model", return_value=True),
        ):
            result = runner.invoke(cli, ["model", "pull", "qwen2.5:14b"], input="y\n")

        assert result.exit_code == 0
        assert config.model == "ollama_chat/qwen2.5:14b"
        mock_save.assert_called_once()

    def test_pull_decline_default_does_not_change_model(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        original_model = config.model
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.cli.save_config"),
            patch("shard.models.pull_ollama_model", return_value=True),
        ):
            result = runner.invoke(cli, ["model", "pull", "phi3.5"], input="n\n")

        assert result.exit_code == 0
        assert config.model == original_model


# ── TestAuthError ─────────────────────────────────────────────────────────────


class TestAuthError:
    """complete() should convert litellm.AuthenticationError to a friendly ModelError."""

    def test_auth_error_raises_model_error(self) -> None:
        from shard.models import complete

        fake_config = MagicMock()
        fake_config.model = "gpt-4o"

        auth_exc = litellm.AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="gpt-4o",
        )

        with (
            patch("shard.models.get_config", return_value=fake_config),
            patch("shard.models.inject_api_keys"),
            patch("litellm.completion", side_effect=auth_exc),
        ):
            with pytest.raises(ModelError) as exc_info:
                complete("hello", model="gpt-4o")

        assert "shard model key" in str(exc_info.value)

    def test_auth_error_message_includes_provider(self) -> None:
        from shard.models import complete

        fake_config = MagicMock()
        fake_config.model = "gpt-4o"

        auth_exc = litellm.AuthenticationError(
            message="Bad key",
            llm_provider="openai",
            model="gpt-4o",
        )

        with (
            patch("shard.models.get_config", return_value=fake_config),
            patch("shard.models.inject_api_keys"),
            patch("litellm.completion", side_effect=auth_exc),
        ):
            with pytest.raises(ModelError) as exc_info:
                complete("test", model="gpt-4o")

        assert "openai" in str(exc_info.value)

    def test_auth_error_message_includes_key_url(self) -> None:
        from shard.models import complete

        fake_config = MagicMock()
        fake_config.model = "gpt-4o"

        auth_exc = litellm.AuthenticationError(
            message="Bad key",
            llm_provider="openai",
            model="gpt-4o",
        )

        with (
            patch("shard.models.get_config", return_value=fake_config),
            patch("shard.models.inject_api_keys"),
            patch("litellm.completion", side_effect=auth_exc),
        ):
            with pytest.raises(ModelError) as exc_info:
                complete("test", model="gpt-4o")

        # The friendly message should link to where the user can get a key.
        error_text = str(exc_info.value)
        assert "platform.openai.com" in error_text or "openai" in error_text

    def test_generic_exception_raises_model_error(self) -> None:
        from shard.models import complete

        fake_config = MagicMock()
        fake_config.model = "ollama_chat/qwen2.5:3b"

        with (
            patch("shard.models.get_config", return_value=fake_config),
            patch("shard.models.inject_api_keys"),
            patch("litellm.completion", side_effect=RuntimeError("connection refused")),
        ):
            with pytest.raises(ModelError) as exc_info:
                complete("hello", model="ollama_chat/qwen2.5:3b")

        assert "connection refused" in str(exc_info.value)

    def test_malformed_response_raises_model_error(self) -> None:
        from shard.models import complete

        fake_config = MagicMock()
        fake_config.model = "ollama_chat/qwen2.5:3b"

        bad_response = MagicMock()
        bad_response.choices = []  # IndexError on choices[0]

        with (
            patch("shard.models.get_config", return_value=fake_config),
            patch("shard.models.inject_api_keys"),
            patch("litellm.completion", return_value=bad_response),
        ):
            with pytest.raises(ModelError) as exc_info:
                complete("hello", model="ollama_chat/qwen2.5:3b")

        assert "Unexpected response" in str(exc_info.value)

    def test_complete_returns_content_on_success(self) -> None:
        from shard.models import complete

        fake_config = MagicMock()
        fake_config.model = "ollama_chat/qwen2.5:3b"

        mock_choice = MagicMock()
        mock_choice.message.content = "The answer is 42."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with (
            patch("shard.models.get_config", return_value=fake_config),
            patch("shard.models.inject_api_keys"),
            patch("litellm.completion", return_value=mock_response),
        ):
            result = complete("What is the answer?", model="ollama_chat/qwen2.5:3b")

        assert result == "The answer is 42."


# ── TestInjectApiKeys ─────────────────────────────────────────────────────────


class TestInjectApiKeys:
    """inject_api_keys() loads keys from config into the process environment.

    Each test uses ``clear=True`` in ``patch.dict`` so that real API keys
    already present in the developer's shell cannot interfere with assertions.
    """

    def test_inject_sets_env_var_from_config(self, tmp_path: Path) -> None:
        from shard.models import inject_api_keys

        config = _make_config(tmp_path)
        config.api_keys["openai"] = "sk-from-config-inject"

        with (
            patch("shard.models.get_config", return_value=config),
            # clear=True gives us a clean slate — no real keys leak in.
            patch.dict(os.environ, {}, clear=True),
        ):
            inject_api_keys()
            result = os.environ.get("OPENAI_API_KEY")

        assert result == "sk-from-config-inject"

    def test_inject_does_not_override_existing_env_var(self, tmp_path: Path) -> None:
        from shard.models import inject_api_keys

        config = _make_config(tmp_path)
        config.api_keys["openai"] = "sk-config-value"

        with (
            patch("shard.models.get_config", return_value=config),
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "sk-already-in-env"},
                clear=True,
            ),
        ):
            inject_api_keys()
            result = os.environ.get("OPENAI_API_KEY")

        # Pre-existing env var must be preserved untouched.
        assert result == "sk-already-in-env"

    def test_inject_sets_multiple_providers(self, tmp_path: Path) -> None:
        from shard.models import inject_api_keys

        config = _make_config(tmp_path)
        config.api_keys["openai"] = "sk-open"
        config.api_keys["anthropic"] = "sk-ant-value"

        with (
            patch("shard.models.get_config", return_value=config),
            patch.dict(os.environ, {}, clear=True),
        ):
            inject_api_keys()
            openai_val = os.environ.get("OPENAI_API_KEY")
            anthropic_val = os.environ.get("ANTHROPIC_API_KEY")

        assert openai_val == "sk-open"
        assert anthropic_val == "sk-ant-value"

    def test_inject_skips_providers_with_no_key(self, tmp_path: Path) -> None:
        from shard.models import inject_api_keys

        config = _make_config(tmp_path)
        # api_keys is empty — nothing should be written to the environment.

        with (
            patch("shard.models.get_config", return_value=config),
            patch.dict(os.environ, {}, clear=True),
        ):
            inject_api_keys()
            has_key = "OPENAI_API_KEY" in os.environ

        assert not has_key


# ── TestDetectProvider ────────────────────────────────────────────────────────


class TestDetectProvider:
    """_detect_provider() maps model strings to provider names."""

    @pytest.mark.parametrize(
        "model_string, expected_provider",
        [
            ("ollama_chat/qwen2.5:3b", "ollama"),
            ("ollama/llama3", "ollama"),
            ("gpt-4o", "openai"),
            ("gpt-5", "openai"),
            ("o1-mini", "openai"),
            ("o3-mini", "openai"),
            ("claude-sonnet-4-20250514", "anthropic"),
            ("groq/llama3-70b", "groq"),
            ("gemini/gemini-2.0-flash", "gemini"),
            ("totally-unknown-model", "unknown"),
        ],
    )
    def test_detect_provider(self, model_string: str, expected_provider: str) -> None:
        from shard.models import _detect_provider

        assert _detect_provider(model_string) == expected_provider


# ── TestDetectAvailableModels ─────────────────────────────────────────────────


class TestDetectAvailableModels:
    """detect_available_models() returns prefixed Ollama model names."""

    def test_returns_prefixed_model_names(self) -> None:
        from shard.models import detect_available_models

        payload = {"models": [{"name": "qwen2.5:3b"}, {"name": "phi3.5"}]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status = MagicMock()

        with patch("shard.models.httpx.get", return_value=mock_resp):
            result = detect_available_models()

        assert "ollama_chat/qwen2.5:3b" in result
        assert "ollama_chat/phi3.5" in result

    def test_returns_empty_list_when_ollama_unreachable(self) -> None:
        import httpx as _httpx

        from shard.models import detect_available_models

        with patch(
            "shard.models.httpx.get",
            side_effect=_httpx.ConnectError("connection refused"),
        ):
            result = detect_available_models()

        assert result == []

    def test_returns_empty_list_on_unexpected_payload(self) -> None:
        from shard.models import detect_available_models

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"unexpected": "structure"}
        mock_resp.raise_for_status = MagicMock()

        with patch("shard.models.httpx.get", return_value=mock_resp):
            result = detect_available_models()

        assert result == []

    def test_skips_entries_without_name_field(self) -> None:
        from shard.models import detect_available_models

        payload = {"models": [{"name": "qwen2.5:3b"}, {"no_name_here": True}]}
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status = MagicMock()

        with patch("shard.models.httpx.get", return_value=mock_resp):
            result = detect_available_models()

        assert len(result) == 1
        assert "ollama_chat/qwen2.5:3b" in result
