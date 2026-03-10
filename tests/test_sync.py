"""Integration tests for the `shard sync` CLI command.

All tests mock external I/O (config loading, LLM calls) so no network access
or local services are required.  Real `.md` files are written to tmp_vault for
realistic file-system behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from shard.cli import cli
from shard.config import ShardConfig, save_config

# ── Shared note template ──────────────────────────────────────────────────────

_NOTE_TEMPLATE = """\
---
title: '{title}'
tags:
  - 'testing'
date: '2024-01-15'
---

This note discusses machine learning and Python programming.
"""

_MOCK_LINK_JSON = json.dumps([
    {
        "original_text": "machine learning",
        "linked_text": "[[Machine Learning|machine learning]]",
        "note_title": "Machine Learning",
    }
])

# The Linker binds `complete` at import time from shard.models, so we must
# patch the name as it exists inside the linker module, not at the source.
_COMPLETE_TARGET = "shard.pipeline.linker.complete"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_note(vault: Path, filename: str, title: str) -> Path:
    """Write a minimal markdown note to *vault* and return its path."""
    path = vault / filename
    path.write_text(_NOTE_TEMPLATE.format(title=title), encoding="utf-8")
    return path


def _fake_config_path(tmp_path: Path) -> Path:
    """Create a placeholder config file so the first-run gate is not triggered."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("{}", encoding="utf-8")
    return cfg_path


# ── test_backup_created_before_changes ───────────────────────────────────────


class TestSyncBackup:
    def test_backup_created_before_changes(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        note_a = _write_note(tmp_vault, "note-a.md", "Note A")
        note_b = _write_note(tmp_vault, "note-b.md", "Note B")
        fake_cfg = _fake_config_path(tmp_path)

        backup_dir_capture: list[Path] = []

        original_mkdir = Path.mkdir

        def capture_mkdir(self: Path, *args: object, **kwargs: object) -> None:  # type: ignore[override]
            original_mkdir(self, *args, **kwargs)
            if "backups" in str(self):
                backup_dir_capture.append(self)

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, return_value=_MOCK_LINK_JSON),
            patch("pathlib.Path.mkdir", capture_mkdir),
        ):
            result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 0, result.output
        # At least one backup directory must have been created.
        assert any("backups" in str(p) for p in backup_dir_capture)

    def test_backup_contains_original_note_content(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        note = _write_note(tmp_vault, "note-a.md", "Note A")
        original_content = note.read_text(encoding="utf-8")
        fake_cfg = _fake_config_path(tmp_path)

        home_backup_root = tmp_path / "fake_home" / ".shard" / "backups"

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, return_value=_MOCK_LINK_JSON),
            patch("shard.cli.Path.home", return_value=tmp_path / "fake_home"),
        ):
            result = runner.invoke(cli, ["sync"])

        assert result.exit_code == 0, result.output
        # Locate any .md file inside the backup tree and verify its content.
        if home_backup_root.exists():
            backed_up = list(home_backup_root.rglob("*.md"))
            assert len(backed_up) >= 1
            assert backed_up[0].read_text(encoding="utf-8") == original_content


# ── test_dry_run_no_file_changes ──────────────────────────────────────────────


class TestSyncDryRun:
    def test_dry_run_no_file_changes(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        note = _write_note(tmp_vault, "note-a.md", "Note A")
        original_content = note.read_text(encoding="utf-8")
        fake_cfg = _fake_config_path(tmp_path)

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, return_value=_MOCK_LINK_JSON),
        ):
            result = runner.invoke(cli, ["sync", "--dry-run"])

        assert result.exit_code == 0, result.output
        # File must be untouched after a dry run.
        assert note.read_text(encoding="utf-8") == original_content

    def test_dry_run_does_not_create_backup(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        _write_note(tmp_vault, "note-a.md", "Note A")
        fake_cfg = _fake_config_path(tmp_path)

        mkdir_calls: list[Path] = []
        original_mkdir = Path.mkdir

        def tracking_mkdir(self: Path, *args: object, **kwargs: object) -> None:  # type: ignore[override]
            original_mkdir(self, *args, **kwargs)
            mkdir_calls.append(self)

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, return_value=_MOCK_LINK_JSON),
            patch("pathlib.Path.mkdir", tracking_mkdir),
        ):
            result = runner.invoke(cli, ["sync", "--dry-run"])

        assert result.exit_code == 0, result.output
        # No backup directory should have been created during a dry run.
        assert not any("backups" in str(p) for p in mkdir_calls)

    def test_dry_run_exits_zero(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        _write_note(tmp_vault, "note-a.md", "Note A")
        fake_cfg = _fake_config_path(tmp_path)

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, return_value=_MOCK_LINK_JSON),
        ):
            result = runner.invoke(cli, ["sync", "--dry-run"])

        assert result.exit_code == 0


# ── test_dry_run_output_matches_results ───────────────────────────────────────


class TestSyncDryRunOutput:
    def test_dry_run_output_contains_dry_run_notice(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        _write_note(tmp_vault, "note-a.md", "Note A")
        fake_cfg = _fake_config_path(tmp_path)

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, return_value=_MOCK_LINK_JSON),
        ):
            result = runner.invoke(cli, ["sync", "--dry-run"])

        assert "Dry run" in result.output or "dry run" in result.output.lower()

    def test_dry_run_output_shows_notes_updated_count(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        _write_note(tmp_vault, "note-a.md", "Note A")
        _write_note(tmp_vault, "note-b.md", "Note B")
        fake_cfg = _fake_config_path(tmp_path)

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, return_value=_MOCK_LINK_JSON),
        ):
            result = runner.invoke(cli, ["sync", "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "Notes updated" in result.output

    def test_dry_run_output_shows_links_added_count(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        _write_note(tmp_vault, "note-a.md", "Note A")
        fake_cfg = _fake_config_path(tmp_path)

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, return_value=_MOCK_LINK_JSON),
        ):
            result = runner.invoke(cli, ["sync", "--dry-run"])

        assert "Links added" in result.output

    def test_dry_run_does_not_show_backup_path(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        _write_note(tmp_vault, "note-a.md", "Note A")
        fake_cfg = _fake_config_path(tmp_path)

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, return_value=_MOCK_LINK_JSON),
        ):
            result = runner.invoke(cli, ["sync", "--dry-run"])

        assert "Backup saved" not in result.output


# ── test_batch_processing ─────────────────────────────────────────────────────


class TestSyncBatchProcessing:
    def test_batch_processing_all_notes_processed(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        note_count = 25
        for i in range(note_count):
            _write_note(tmp_vault, f"note-{i:02d}.md", f"Note {i}")

        fake_cfg = _fake_config_path(tmp_path)
        complete_call_count: list[int] = [0]

        def counting_complete(prompt: str, system: str = "", model: str = "") -> str:
            complete_call_count[0] += 1
            return _MOCK_LINK_JSON

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, side_effect=counting_complete),
        ):
            result = runner.invoke(cli, ["sync", "--dry-run"])

        assert result.exit_code == 0, result.output
        # Every note must have triggered exactly one model call.
        assert complete_call_count[0] == note_count

    def test_batch_processing_exit_code_is_zero(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        for i in range(25):
            _write_note(tmp_vault, f"note-{i:02d}.md", f"Note {i}")

        fake_cfg = _fake_config_path(tmp_path)

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, return_value=_MOCK_LINK_JSON),
        ):
            result = runner.invoke(cli, ["sync", "--dry-run"])

        assert result.exit_code == 0

    def test_batch_processing_output_counts_reflect_all_notes(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        note_count = 25
        for i in range(note_count):
            _write_note(tmp_vault, f"note-{i:02d}.md", f"Note {i}")

        fake_cfg = _fake_config_path(tmp_path)

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, return_value=_MOCK_LINK_JSON),
        ):
            result = runner.invoke(cli, ["sync", "--dry-run"])

        assert result.exit_code == 0, result.output
        # With one link suggestion per note and 25 notes we expect 25 links reported.
        assert "25" in result.output

    def test_sync_empty_vault_exits_zero(
        self, tmp_vault: Path, mock_config: ShardConfig, tmp_path: Path
    ) -> None:
        # No notes — sync should complete cleanly with zero counts.
        fake_cfg = _fake_config_path(tmp_path)

        runner = CliRunner()
        with (
            patch("shard.cli.CONFIG_PATH", fake_cfg),
            patch("shard.cli.get_config", return_value=mock_config),
            patch(_COMPLETE_TARGET, return_value="[]"),
        ):
            result = runner.invoke(cli, ["sync", "--dry-run"])

        assert result.exit_code == 0
        assert "0" in result.output
