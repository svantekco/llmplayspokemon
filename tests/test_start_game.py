from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

from pokemon_agent.main import apply_runtime_environment, build_main_args, resolve_planner, session_is_resumable


def test_start_game_uses_resume_when_checkpoint_exists(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "checkpoint.json").write_text(json.dumps({"completed_turns": 2}), encoding="utf-8")
    (session_dir / "emulator.state").write_bytes(b"state")

    args = Namespace(
        mode="mock",
        rom="game.gb",
        turns=3,
        once=False,
        planner="fallback",
        session_dir=str(session_dir),
        fresh=False,
        log_mode="compact",
        window=None,
        headless=False,
        checkpoint_dir=None,
        resume=None,
        continuous=False,
    )

    main_args = build_main_args(args)

    assert main_args.resume == str(session_dir.resolve())
    assert main_args.checkpoint_dir == str(session_dir.resolve())
    assert main_args.continuous is True
    assert main_args.log_mode == "compact"


def test_start_game_does_not_resume_without_emulator_state(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "checkpoint.json").write_text(json.dumps({"completed_turns": 2}), encoding="utf-8")

    args = Namespace(
        mode="mock",
        rom="game.gb",
        turns=3,
        once=False,
        planner="fallback",
        session_dir=str(session_dir),
        fresh=False,
        log_mode="compact",
        window=None,
        headless=False,
        checkpoint_dir=None,
        resume=None,
        continuous=False,
    )

    main_args = build_main_args(args)

    assert main_args.resume is None


def test_session_is_resumable_requires_both_checkpoint_and_emulator_state(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    assert session_is_resumable(session_dir) is False

    (session_dir / "checkpoint.json").write_text("{}", encoding="utf-8")
    assert session_is_resumable(session_dir) is False

    (session_dir / "emulator.state").write_bytes(b"state")
    assert session_is_resumable(session_dir) is True


def test_start_game_clears_session_when_fresh(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "checkpoint.json").write_text("{}", encoding="utf-8")

    args = Namespace(
        mode="mock",
        rom="game.gb",
        turns=3,
        once=False,
        planner="fallback",
        session_dir=str(session_dir),
        fresh=True,
        log_mode="compact",
        window=None,
        headless=False,
        checkpoint_dir=None,
        resume=None,
        continuous=False,
    )

    main_args = build_main_args(args)

    assert main_args.resume is None
    assert (session_dir / "checkpoint.json").exists() is False


def test_start_game_defaults_to_visible_pyboy_window(monkeypatch):
    monkeypatch.delenv("PYBOY_WINDOW", raising=False)
    args = Namespace(
        mode="pyboy",
        rom="game.gb",
        turns=3,
        once=False,
        planner="fallback",
        session_dir=".sessions/default",
        fresh=False,
        window=None,
        headless=False,
        log_mode="compact",
        checkpoint_dir=None,
        resume=None,
        continuous=False,
    )

    window = apply_runtime_environment(args)

    assert window == "SDL2"
    assert os.environ["PYBOY_WINDOW"] == "SDL2"


def test_start_game_auto_planner_uses_llm_when_key_exists(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    args = Namespace(
        mode="pyboy",
        rom="game.gb",
        turns=3,
        once=False,
        planner="auto",
        session_dir=".sessions/default",
        fresh=False,
        window=None,
        headless=False,
        log_mode="compact",
        checkpoint_dir=None,
        resume=None,
        continuous=False,
    )

    assert resolve_planner(args) == "llm"
