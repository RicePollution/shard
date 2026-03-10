"""Tests for shard.pipeline.learner — style analysis, JSON parsing, profile I/O,
and the CLI learn --force flag.

All LLM calls are mocked so no network access or local model is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from click.testing import CliRunner

from shard.pipeline import LearnError
from shard.pipeline.learner import (
    QUICK_SAMPLE_SIZE,
    Learner,
    StyleProfile,
    _parse_json_response,
    load_style_profile,
    save_style_profile,
)

# ── Shared test data ──────────────────────────────────────────────────────────

_PASS1_JSON = json.dumps(
    {
        "title_format": "Test Note Title",
        "heading_levels_used": ["##", "###"],
        "first_section_heading": "Overview",
        "bullet_style": "dash",
        "has_frontmatter": True,
        "frontmatter_fields": ["tags", "date"],
        "tag_format": "#lowercase-hyphen",
        "tag_count": 3,
        "tag_examples": ["#python", "#testing", "#dev"],
        "avg_section_length": "medium(4-8)",
        "uses_bold": True,
        "uses_italic": False,
        "uses_callouts": False,
        "callout_examples": [],
        "uses_code_blocks": True,
        "link_style": "wikilink [[]]",
        "opens_with": "frontmatter",
        "closes_with": "tags",
        "word_count": 320,
        "sentence_examples": [
            "This is a test sentence.",
            "Another example here.",
        ],
    }
)

_PASS2_JSON = json.dumps(
    {
        "style_rules": "Always use ## headers. Use dash bullets.",
        "template": (
            "---\ntags:\ndate:\n---\n\n## Overview\n\n## Notes\n\n## Related\n"
        ),
        "fingerprints": [
            "Always opens with frontmatter",
            "Uses ## Overview as first heading",
            "Tags: #lowercase-hyphen",
            "Ends with ## Related",
            "3-5 tags per note",
        ],
        "frontmatter_template": "---\ntags:\ndate:\n---",
        "heading_order": ["Overview", "Notes", "Related"],
        "tag_format": "#lowercase-hyphen",
        "avg_word_count": 320,
        "tone_examples": [
            "This is a test sentence.",
            "Another example here.",
        ],
    }
)


def _make_notes(count: int) -> list[str]:
    """Return *count* minimal note strings."""
    return [f"# Note {i}\n\nContent for note number {i}." for i in range(count)]


def _make_style_profile(**overrides: object) -> StyleProfile:
    """Return a fully-populated StyleProfile, optionally overriding fields."""
    defaults: dict[str, object] = dict(
        style_rules="Always use ## headers. Use dash bullets.",
        template="---\ntags:\ndate:\n---\n\n## Overview\n\n## Notes\n\n## Related\n",
        fingerprints=["Always opens with frontmatter", "Tags: #lowercase-hyphen"],
        frontmatter_template="---\ntags:\ndate:\n---",
        heading_order=["Overview", "Notes", "Related"],
        tag_format="#lowercase-hyphen",
        avg_word_count=320,
        tone_examples=["This is a test sentence.", "Another example here."],
        analyzed_at="2026-03-09T00:00:00+00:00",
        notes_sampled=5,
    )
    defaults.update(overrides)
    return StyleProfile(**defaults)  # type: ignore[arg-type]


# ── Learner.analyze ───────────────────────────────────────────────────────────


class TestLearnerAnalyze:
    def test_analyze_returns_style_profile(self) -> None:
        """Two-pass analysis with mocked LLM returns a valid StyleProfile."""
        learner = Learner()
        notes = _make_notes(5)

        with patch("shard.pipeline.learner.complete") as mock_complete:
            # Pass 1 returns per-note JSON; pass 2 returns synthesis JSON.
            mock_complete.side_effect = [_PASS1_JSON] * 5 + [_PASS2_JSON]
            profile = learner.analyze(notes)

        assert isinstance(profile, StyleProfile)
        assert profile.style_rules == "Always use ## headers. Use dash bullets."
        assert profile.tag_format == "#lowercase-hyphen"
        assert profile.avg_word_count == 320
        assert profile.fingerprints == [
            "Always opens with frontmatter",
            "Uses ## Overview as first heading",
            "Tags: #lowercase-hyphen",
            "Ends with ## Related",
            "3-5 tags per note",
        ]
        assert profile.heading_order == ["Overview", "Notes", "Related"]
        assert profile.notes_sampled == 5

    def test_analyze_rejects_fewer_than_5_notes(self) -> None:
        """Passing fewer than 5 notes raises LearnError immediately."""
        learner = Learner()

        with pytest.raises(LearnError, match="at least 5"):
            learner.analyze(_make_notes(3))

    def test_analyze_rejects_zero_notes(self) -> None:
        """Empty note list raises LearnError."""
        learner = Learner()

        with pytest.raises(LearnError):
            learner.analyze([])

    def test_analyze_samples_max_20(self) -> None:
        """When more than 20 notes are provided, only 20 LLM calls are made in pass 1."""
        learner = Learner()
        notes = _make_notes(30)

        with patch("shard.pipeline.learner.complete") as mock_complete:
            # 20 pass-1 calls + 1 pass-2 call = 21 total
            mock_complete.side_effect = [_PASS1_JSON] * 20 + [_PASS2_JSON]
            learner.analyze(notes)

        # Total calls: 20 (pass 1) + 1 (pass 2) = 21
        assert mock_complete.call_count == 21

    def test_analyze_exactly_5_notes_does_not_sample(self) -> None:
        """Exactly 5 notes are all used without random sampling."""
        learner = Learner()
        notes = _make_notes(5)

        with patch("shard.pipeline.learner.complete") as mock_complete:
            mock_complete.side_effect = [_PASS1_JSON] * 5 + [_PASS2_JSON]
            profile = learner.analyze(notes)

        assert profile.notes_sampled == 5
        assert mock_complete.call_count == 6  # 5 + 1

    def test_analyze_raises_learn_error_when_complete_fails(self) -> None:
        """A model failure during pass 1 is wrapped into LearnError."""
        learner = Learner()
        notes = _make_notes(5)

        with patch(
            "shard.pipeline.learner.complete",
            side_effect=RuntimeError("connection refused"),
        ):
            with pytest.raises(LearnError, match="Model call failed"):
                learner.analyze(notes)


# ── _parse_json_response ──────────────────────────────────────────────────────


class TestParseJsonResponse:
    def test_parses_bare_json_object(self) -> None:
        raw = '{"key": "value", "num": 42}'
        result = _parse_json_response(raw)
        assert result == {"key": "value", "num": 42}

    def test_strips_json_code_fences(self) -> None:
        raw = '```json\n{"key": "fenced"}\n```'
        result = _parse_json_response(raw)
        assert result == {"key": "fenced"}

    def test_strips_plain_code_fences(self) -> None:
        raw = '```\n{"key": "plain-fence"}\n```'
        result = _parse_json_response(raw)
        assert result == {"key": "plain-fence"}

    def test_extracts_json_from_surrounding_prose(self) -> None:
        raw = 'Here is the result:\n{"answer": true}\nDone.'
        result = _parse_json_response(raw)
        assert result == {"answer": True}

    def test_handles_leading_and_trailing_whitespace(self) -> None:
        raw = '   \n  {"x": 1}  \n   '
        result = _parse_json_response(raw)
        assert result == {"x": 1}

    def test_raises_learn_error_on_invalid_json(self) -> None:
        with pytest.raises(LearnError, match="Failed to parse"):
            _parse_json_response("this is not json at all !!!")

    def test_raises_learn_error_on_empty_string(self) -> None:
        with pytest.raises(LearnError):
            _parse_json_response("")

    def test_returns_list_for_json_array_input(self) -> None:
        # _parse_json_response looks for { } bounds; when absent the full text
        # is passed to json.loads, which happily parses a JSON array and returns
        # it as-is.  The function makes no assertion about the return type being
        # a dict, so callers that pass an array payload receive a list back.
        result = _parse_json_response("[1, 2, 3]")
        assert result == [1, 2, 3]  # type: ignore[comparison-overlap]


# ── save_style_profile / load_style_profile ───────────────────────────────────


class TestStyleProfilePersistence:
    def test_save_and_load_style_profile(self, tmp_path: Path) -> None:
        """A saved StyleProfile round-trips exactly through load_style_profile."""
        profile = _make_style_profile()
        dest = tmp_path / "style.json"

        save_style_profile(profile, dest)
        loaded = load_style_profile(dest)

        assert loaded is not None
        assert loaded.style_rules == profile.style_rules
        assert loaded.template == profile.template
        assert loaded.fingerprints == profile.fingerprints
        assert loaded.frontmatter_template == profile.frontmatter_template
        assert loaded.heading_order == profile.heading_order
        assert loaded.tag_format == profile.tag_format
        assert loaded.avg_word_count == profile.avg_word_count
        assert loaded.tone_examples == profile.tone_examples
        assert loaded.analyzed_at == profile.analyzed_at
        assert loaded.notes_sampled == profile.notes_sampled

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        """save_style_profile creates missing parent directories."""
        profile = _make_style_profile()
        dest = tmp_path / "deep" / "nested" / "style.json"

        save_style_profile(profile, dest)

        assert dest.exists()

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        """The saved file is valid UTF-8 JSON."""
        profile = _make_style_profile()
        dest = tmp_path / "style.json"

        save_style_profile(profile, dest)
        raw = json.loads(dest.read_text(encoding="utf-8"))

        assert raw["style_rules"] == profile.style_rules
        assert raw["avg_word_count"] == profile.avg_word_count

    def test_load_style_profile_returns_none_when_missing(self, tmp_path: Path) -> None:
        """load_style_profile returns None when the file does not exist."""
        missing = tmp_path / "nonexistent" / "style.json"
        result = load_style_profile(missing)
        assert result is None

    def test_load_style_profile_raises_on_corrupt_json(self, tmp_path: Path) -> None:
        """load_style_profile raises LearnError when the file contains invalid JSON."""
        corrupt = tmp_path / "style.json"
        corrupt.write_text("{ not valid json }", encoding="utf-8")

        with pytest.raises(LearnError):
            load_style_profile(corrupt)


# ── CLI learn --force ─────────────────────────────────────────────────────────


def _cli_learn_patches(tmp_path: Path, existing_style: Path) -> dict:
    """Return a dict of patch targets shared by CLI learn tests."""
    from shard.config import ShardConfig

    vault = tmp_path / "vault"
    vault.mkdir(exist_ok=True)
    config = ShardConfig(
        vault_path=vault,
        chroma_path=tmp_path / ".chroma",
        model="ollama_chat/qwen2.5:3b",
    )
    fake_config_path = tmp_path / "config.json"
    fake_config_path.write_text("{}", encoding="utf-8")
    return {
        "config": config,
        "fake_config_path": fake_config_path,
    }


class TestCliLearnForce:
    def test_without_force_skips_reanalysis_when_profile_exists(
        self, tmp_path: Path
    ) -> None:
        """Without --force, an existing style profile causes an early return with hint."""
        from shard.cli import cli

        runner = CliRunner()
        info = _cli_learn_patches(tmp_path, tmp_path / "style.json")

        existing_style = tmp_path / "style.json"
        save_style_profile(_make_style_profile(), existing_style)

        mock_pass1 = MagicMock()
        mock_pass2 = MagicMock()

        with (
            patch("shard.cli.CONFIG_PATH", info["fake_config_path"]),
            patch("shard.cli.STYLE_PROFILE_PATH", existing_style),
            patch("shard.cli.get_config", return_value=info["config"]),
            patch("shard.pipeline.learner.Learner._pass1_extract", mock_pass1),
            patch("shard.pipeline.learner.Learner._pass2_synthesize", mock_pass2),
        ):
            result = runner.invoke(cli, ["learn"])

        # Neither pass should run — CLI returns early.
        mock_pass1.assert_not_called()
        mock_pass2.assert_not_called()
        assert result.exit_code == 0
        assert "--force" in result.output

    def test_force_flag_bypasses_existing_profile_check(self, tmp_path: Path) -> None:
        """With --force, the early-return branch is skipped and pass 1 runs."""
        from shard.cli import cli

        runner = CliRunner()
        info = _cli_learn_patches(tmp_path, tmp_path / "style.json")

        existing_style = tmp_path / "style.json"
        save_style_profile(_make_style_profile(), existing_style)

        note_paths = [tmp_path / f"note{i}.md" for i in range(6)]
        pass1_data = [json.loads(_PASS1_JSON)] * 6
        mock_profile = _make_style_profile()

        with (
            patch("shard.cli.CONFIG_PATH", info["fake_config_path"]),
            patch("shard.cli.STYLE_PROFILE_PATH", existing_style),
            patch("shard.cli.get_config", return_value=info["config"]),
            patch("shard.vault.walk_vault", return_value=note_paths),
            patch("shard.vault.read_note", return_value="# Note\n\nContent"),
            patch(
                "shard.pipeline.learner.Learner._pass1_extract",
                return_value=pass1_data,
            ) as mock_pass1,
            patch(
                "shard.pipeline.learner.Learner._pass2_synthesize",
                return_value=mock_profile,
            ),
            patch("shard.pipeline.learner.save_style_profile"),
        ):
            result = runner.invoke(cli, ["learn", "--force"])

        # Pass 1 must have been called — proof that analysis ran.
        mock_pass1.assert_called_once()
        assert result.exit_code == 0

    def test_force_flag_calls_pass2_synthesize(self, tmp_path: Path) -> None:
        """With --force, pass 2 synthesis runs and its result is saved."""
        from shard.cli import cli

        runner = CliRunner()
        info = _cli_learn_patches(tmp_path, tmp_path / "style.json")

        existing_style = tmp_path / "style.json"
        save_style_profile(_make_style_profile(), existing_style)

        note_paths = [tmp_path / f"note{i}.md" for i in range(6)]
        pass1_data = [json.loads(_PASS1_JSON)] * 6
        mock_profile = _make_style_profile()

        with (
            patch("shard.cli.CONFIG_PATH", info["fake_config_path"]),
            patch("shard.cli.STYLE_PROFILE_PATH", existing_style),
            patch("shard.cli.get_config", return_value=info["config"]),
            patch("shard.vault.walk_vault", return_value=note_paths),
            patch("shard.vault.read_note", return_value="# Note\n\nContent"),
            patch(
                "shard.pipeline.learner.Learner._pass1_extract",
                return_value=pass1_data,
            ),
            patch(
                "shard.pipeline.learner.Learner._pass2_synthesize",
                return_value=mock_profile,
            ) as mock_pass2,
            patch("shard.pipeline.learner.save_style_profile") as mock_save,
        ):
            runner.invoke(cli, ["learn", "--force"])

        mock_pass2.assert_called_once()
        mock_save.assert_called_once()


# ── Learner depth modes ──────────────────────────────────────────────────────


class TestLearnerQuickDepth:
    def test_quick_makes_exactly_one_api_call(self) -> None:
        """Quick depth skips Pass 1 and makes a single synthesis call."""
        learner = Learner()
        notes = _make_notes(10)

        with patch("shard.pipeline.learner.complete") as mock_complete:
            # Quick mode: 1 call total (direct synthesis)
            mock_complete.return_value = _PASS2_JSON
            learner.analyze(notes, depth="quick")

        assert mock_complete.call_count == 1

    def test_quick_samples_5_notes_max(self) -> None:
        """Quick depth samples at most QUICK_SAMPLE_SIZE (5) notes."""
        learner = Learner()
        notes = _make_notes(20)

        captured_prompt: list[str] = []

        def capture(prompt: str, **kwargs: object) -> str:
            captured_prompt.append(prompt)
            return _PASS2_JSON

        with patch("shard.pipeline.learner.complete", side_effect=capture):
            learner.analyze(notes, depth="quick")

        # The prompt should mention 5 notes (QUICK_SAMPLE_SIZE)
        assert "5" in captured_prompt[0]

    def test_quick_returns_valid_style_profile(self) -> None:
        """Quick depth returns a complete StyleProfile."""
        learner = Learner()
        notes = _make_notes(5)

        with patch("shard.pipeline.learner.complete", return_value=_PASS2_JSON):
            profile = learner.analyze(notes, depth="quick")

        assert isinstance(profile, StyleProfile)
        assert profile.notes_sampled == 5

    def test_quick_with_fewer_than_5_notes_uses_all(self) -> None:
        """Quick depth with exactly 5 notes uses all of them."""
        learner = Learner()
        notes = _make_notes(5)

        with patch("shard.pipeline.learner.complete", return_value=_PASS2_JSON):
            profile = learner.analyze(notes, depth="quick")

        assert profile.notes_sampled == 5

    def test_quick_raises_on_model_failure(self) -> None:
        """Quick depth wraps model errors in LearnError."""
        learner = Learner()
        notes = _make_notes(5)

        with patch(
            "shard.pipeline.learner.complete",
            side_effect=RuntimeError("connection refused"),
        ):
            with pytest.raises(LearnError, match="quick synthesis"):
                learner.analyze(notes, depth="quick")


class TestLearnerNormalDepth:
    def test_normal_is_default(self) -> None:
        """Normal depth is used when depth is not specified."""
        learner = Learner()
        notes = _make_notes(5)

        with patch("shard.pipeline.learner.complete") as mock_complete:
            mock_complete.side_effect = [_PASS1_JSON] * 5 + [_PASS2_JSON]
            learner.analyze(notes)  # no depth kwarg

        # 5 pass-1 + 1 pass-2 = 6 calls
        assert mock_complete.call_count == 6

    def test_normal_samples_max_20(self) -> None:
        """Normal depth caps at MAX_SAMPLE_SIZE (20) notes for pass 1."""
        learner = Learner()
        notes = _make_notes(30)

        with patch("shard.pipeline.learner.complete") as mock_complete:
            mock_complete.side_effect = [_PASS1_JSON] * 20 + [_PASS2_JSON]
            learner.analyze(notes, depth="normal")

        assert mock_complete.call_count == 21


class TestLearnerDeepDepth:
    def test_deep_uses_all_notes(self) -> None:
        """Deep depth processes ALL notes without sampling cap."""
        learner = Learner()
        notes = _make_notes(30)

        with patch("shard.pipeline.learner.complete") as mock_complete:
            # 30 pass-1 calls + 1 pass-2 call = 31 total
            mock_complete.side_effect = [_PASS1_JSON] * 30 + [_PASS2_JSON]
            learner.analyze(notes, depth="deep")

        assert mock_complete.call_count == 31

    def test_deep_with_small_vault(self) -> None:
        """Deep depth on a small vault processes all notes."""
        learner = Learner()
        notes = _make_notes(7)

        with patch("shard.pipeline.learner.complete") as mock_complete:
            mock_complete.side_effect = [_PASS1_JSON] * 7 + [_PASS2_JSON]
            profile = learner.analyze(notes, depth="deep")

        assert mock_complete.call_count == 8
        assert profile.notes_sampled == 7

    def test_deep_returns_valid_profile(self) -> None:
        """Deep depth returns a StyleProfile with correct notes_sampled."""
        learner = Learner()
        notes = _make_notes(10)

        with patch("shard.pipeline.learner.complete") as mock_complete:
            mock_complete.side_effect = [_PASS1_JSON] * 10 + [_PASS2_JSON]
            profile = learner.analyze(notes, depth="deep")

        assert isinstance(profile, StyleProfile)
        assert profile.notes_sampled == 10


# ── CLI learn --depth ────────────────────────────────────────────────────────


class TestCliLearnDepth:
    def test_quick_depth_via_cli(self, tmp_path: Path) -> None:
        """CLI learn --depth quick calls analyze with depth='quick'."""
        from shard.cli import cli
        from shard.config import ShardConfig

        runner = CliRunner()

        vault = tmp_path / "vault"
        vault.mkdir()
        config = ShardConfig(
            vault_path=vault,
            chroma_path=tmp_path / ".chroma",
            model="ollama_chat/qwen2.5:3b",
        )
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        note_paths = [tmp_path / f"note{i}.md" for i in range(6)]
        mock_profile = _make_style_profile()

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.STYLE_PROFILE_PATH", tmp_path / "style.json"),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.vault.walk_vault", return_value=note_paths),
            patch("shard.vault.read_note", return_value="# Note\n\nContent"),
            patch(
                "shard.pipeline.learner.Learner.analyze",
                return_value=mock_profile,
            ) as mock_analyze,
            patch("shard.pipeline.learner.save_style_profile"),
        ):
            result = runner.invoke(cli, ["learn", "--force", "--depth", "quick"])

        mock_analyze.assert_called_once()
        call_kwargs = mock_analyze.call_args
        # Check depth='quick' was passed
        assert call_kwargs[1].get("depth") == "quick" or (
            len(call_kwargs[0]) > 1 and call_kwargs[0][1] == "quick"
        )
        assert result.exit_code == 0

    def test_deep_depth_shows_warning_for_large_vault(self, tmp_path: Path) -> None:
        """CLI learn --depth deep shows confirmation prompt for 50+ notes."""
        from shard.cli import cli
        from shard.config import ShardConfig

        runner = CliRunner()

        vault = tmp_path / "vault"
        vault.mkdir()
        config = ShardConfig(
            vault_path=vault,
            chroma_path=tmp_path / ".chroma",
            model="ollama_chat/qwen2.5:3b",
        )
        fake_config_path = tmp_path / "config.json"
        fake_config_path.write_text("{}", encoding="utf-8")

        note_paths = [tmp_path / f"note{i}.md" for i in range(60)]

        with (
            patch("shard.cli.CONFIG_PATH", fake_config_path),
            patch("shard.cli.STYLE_PROFILE_PATH", tmp_path / "style.json"),
            patch("shard.cli.get_config", return_value=config),
            patch("shard.vault.walk_vault", return_value=note_paths),
            patch("shard.vault.read_note", return_value="# Note\n\nContent"),
        ):
            # User declines the confirmation prompt
            result = runner.invoke(cli, ["learn", "--force", "--depth", "deep"], input="n\n")

        assert result.exit_code == 0
        assert "Aborted" in result.output
