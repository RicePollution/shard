"""Tests for shard.cli using click.testing.CliRunner.

All tests mock out external I/O (config loading, vault access, ChromaDB,
LLM calls) so that no network access or local services are required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from shard.cli import cli
from shard.config import ShardConfig
from shard.pipeline import ConfigError

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_config(tmp_path: Path) -> ShardConfig:
    vault = tmp_path / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    return ShardConfig(
        vault_path=vault,
        chroma_path=tmp_path / ".chroma",
        model="ollama_chat/qwen2.5:3b",
    )


# ── shard --help ──────────────────────────────────────────────────────────────


class TestCliHelp:
    def test_help_exits_with_zero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        # CONFIG_PATH must exist so the first-run wizard is not triggered.
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")  # placeholder

        with patch("shard.cli.CONFIG_PATH", fake_config_path):
            result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0

    def test_help_output_contains_group_description(self, tmp_path: Path) -> None:
        runner = CliRunner()
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        with patch("shard.cli.CONFIG_PATH", fake_config_path):
            result = runner.invoke(cli, ["--help"])

        assert "Shard" in result.output

    def test_help_lists_subcommands(self, tmp_path: Path) -> None:
        runner = CliRunner()
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        with patch("shard.cli.CONFIG_PATH", fake_config_path):
            result = runner.invoke(cli, ["--help"])

        # At least the canonical sub-commands must appear in the help text.
        for cmd in ("add", "list", "ask", "index"):
            assert cmd in result.output

    def test_add_subcommand_help(self, tmp_path: Path) -> None:
        runner = CliRunner()
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        with patch("shard.cli.CONFIG_PATH", fake_config_path):
            result = runner.invoke(cli, ["add", "--help"])

        assert result.exit_code == 0
        assert "INPUT" in result.output

    def test_list_subcommand_help(self, tmp_path: Path) -> None:
        runner = CliRunner()
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        with patch("shard.cli.CONFIG_PATH", fake_config_path):
            result = runner.invoke(cli, ["list", "--help"])

        assert result.exit_code == 0


# ── shard list ────────────────────────────────────────────────────────────────


class TestCliList:
    def test_list_empty_vault_prints_no_notes_message(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        # list_shards is imported inside the list_ command body, so patch at
        # its canonical module location: shard.vault.list_shards.
        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.vault.list_shards", return_value=[]),
        ):
            result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0
        # When the vault is empty the CLI prints a "no shard notes" message to stderr;
        # CliRunner merges stderr into output by default when mix_stderr=True (the default).
        assert "No shard notes found" in result.output

    def test_list_empty_vault_does_not_raise(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.vault.list_shards", return_value=[]),
        ):
            result = runner.invoke(cli, ["list"])

        assert result.exception is None

    def test_list_with_notes_exits_zero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        # Create a minimal .md file in the shards directory
        shards_dir = config.vault_path / "Imported" / "Shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        note_path = shards_dir / "my-note.md"
        note_path.write_text(
            "---\ntitle: 'My Note'\ntags:\n  - 'python'\ndate: '2024-01-01'\n"
            "source: 'https://example.com'\nsource_type: 'url'\nsummary: 'A note.'\n---\n"
            "\nBody content.",
            encoding="utf-8",
        )

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.vault.list_shards", return_value=[note_path]),
        ):
            result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0

    def test_list_with_notes_shows_title_in_output(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        shards_dir = config.vault_path / "Imported" / "Shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        note_path = shards_dir / "my-note.md"
        note_path.write_text(
            "---\ntitle: 'My Unique Note Title'\ntags: []\ndate: '2024-06-15'\n"
            "source: 'text'\nsource_type: 'text'\nsummary: ''\n---\n\nBody.",
            encoding="utf-8",
        )

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.vault.list_shards", return_value=[note_path]),
        ):
            result = runner.invoke(cli, ["list"])

        assert "My Unique Note Title" in result.output

    def test_list_propagates_config_error_message(self, tmp_path: Path) -> None:
        runner = CliRunner()
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.get_config", side_effect=ConfigError("Config missing")),
        ):
            result = runner.invoke(cli, ["list"])

        assert result.exit_code == 1
        assert "Config missing" in result.output


# ── first-run gate ────────────────────────────────────────────────────────────


class TestFirstRunGate:
    def test_first_run_setup_triggered_when_config_missing(self, tmp_path: Path) -> None:
        runner = CliRunner()
        # Point CONFIG_PATH to a path that does not exist
        nonexistent_config = tmp_path / "no-config.json"
        config = _make_config(tmp_path)
        mock_setup = MagicMock(return_value=config)

        with (
            patch("shard.cli.CONFIG_PATH", nonexistent_config),
            patch("shard.cli.first_run_setup", mock_setup),
            # Mock get_config and list_shards so the list command completes cleanly
            # after setup so we can observe that the group callback ran.
            patch("shard.cli.get_config", return_value=config),
            patch("shard.vault.list_shards", return_value=[]),
        ):
            # Use a real subcommand (not --help) so the group callback fires.
            runner.invoke(cli, ["list"])

        mock_setup.assert_called_once()

    def test_first_run_setup_not_triggered_when_config_exists(self, tmp_path: Path) -> None:
        runner = CliRunner()
        existing_config = tmp_path / "config.json"
        existing_config.write_text("{}", encoding="utf-8")
        config = _make_config(tmp_path)
        mock_setup = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", existing_config),
            patch("shard.cli.first_run_setup", mock_setup),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.vault.list_shards", return_value=[]),
        ):
            runner.invoke(cli, ["list"])

        mock_setup.assert_not_called()


# ── shard ask (smoke test with mocks) ─────────────────────────────────────────


class TestCliAsk:
    def test_ask_outputs_answer(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        from shard.search import AskResult

        mock_result = AskResult(answer="The answer is 42.", sources=[])

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.search.ask", return_value=mock_result),
        ):
            result = runner.invoke(cli, ["ask", "What is the meaning of life?"])

        assert result.exit_code == 0
        assert "The answer is 42." in result.output

    def test_ask_with_sources_shows_sources_table(self, tmp_path: Path) -> None:
        runner = CliRunner()
        config = _make_config(tmp_path)
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        from shard.search import AskResult

        mock_result = AskResult(
            answer="Answer from notes.",
            sources=[
                {
                    "title": "Source Note Title",
                    "path": "/vault/note.md",
                    "relevance_score": 0.9,
                }
            ],
        )

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.search.ask", return_value=mock_result),
        ):
            result = runner.invoke(cli, ["ask", "My question"])

        assert result.exit_code == 0
        assert "Source Note Title" in result.output
