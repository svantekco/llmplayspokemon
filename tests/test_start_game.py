from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

from pokemon_agent.main import (
    apply_default_mode,
    apply_runtime_environment,
    build_main_args,
    clear_pyboy_rom_sidecars,
    resolve_planner,
    session_is_resumable,
)


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


def test_start_game_clears_pyboy_rom_sidecars_when_fresh(tmp_path: Path):
    rom_path = tmp_path / "pokemon.gb"
    rom_path.write_bytes(b"rom")
    ram_path = rom_path.with_suffix(".gb.ram")
    ram_path.write_bytes(b"ram")
    sav_path = rom_path.with_suffix(".gb.sav")
    sav_path.write_bytes(b"sav")

    args = Namespace(
        mode="pyboy",
        rom=str(rom_path),
        turns=3,
        once=False,
        planner="fallback",
        session_dir=str(tmp_path / "session"),
        fresh=True,
        log_mode="compact",
        window=None,
        headless=False,
        checkpoint_dir=None,
        resume=None,
        continuous=False,
    )

    main_args = build_main_args(args)

    assert main_args.rom == str(rom_path.resolve())
    assert ram_path.exists() is False
    assert sav_path.exists() is False


def test_start_game_defaults_debug_overlay_dir_inside_session(tmp_path: Path):
    session_dir = tmp_path / "session"

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
        debug_overlay=True,
        debug_overlay_dir=None,
    )

    main_args = build_main_args(args)

    assert main_args.debug_overlay_dir == str((session_dir.resolve() / "debug_overlay"))


def test_clear_pyboy_rom_sidecars_ignores_missing_files(tmp_path: Path):
    rom_path = tmp_path / "pokemon.gb"
    rom_path.write_bytes(b"rom")

    clear_pyboy_rom_sidecars(rom_path)

    assert rom_path.exists()


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


def test_start_game_launcher_defaults_to_pyboy_mode():
    parsed = apply_default_mode(["--session-dir", ".sessions/default", "--planner", "llm", "--fresh"], default_mode="pyboy")

    assert parsed[:2] == ["--mode", "pyboy"]


def test_start_game_launcher_preserves_explicit_mode():
    parsed = apply_default_mode(["--mode=mock", "--session-dir", ".sessions/mock"], default_mode="pyboy")

    assert parsed == ["--mode=mock", "--session-dir", ".sessions/mock"]
