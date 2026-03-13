from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pokemon_agent.main import create_parser, run_args


def create_start_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start or resume a Pokemon Red PyBoy session using the local agent engine."
    )
    parser.add_argument("--mode", choices=["pyboy", "mock"], default="pyboy")
    parser.add_argument("--rom", default="game.gb")
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--once", action="store_true", help="Run a finite batch instead of continuous watch mode.")
    parser.add_argument("--planner", choices=["auto", "llm", "fallback"], default="auto")
    parser.add_argument("--session-dir", default=".sessions/default")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--window", choices=["SDL2", "OpenGL", "GLFW", "null"], default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--log-mode", choices=["dashboard", "compact", "verbose", "quiet"], default="dashboard")
    return parser


def build_main_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = create_parser()
    main_args = parser.parse_args([])
    session_dir = (REPO_ROOT / args.session_dir).resolve()
    checkpoint_path = session_dir / "checkpoint.json"

    if args.eval:
        main_args.eval = True
        return main_args

    if args.fresh and session_dir.exists():
        shutil.rmtree(session_dir)

    session_dir.mkdir(parents=True, exist_ok=True)

    main_args.mode = args.mode
    main_args.rom = str((REPO_ROOT / args.rom).resolve()) if args.mode == "pyboy" else args.rom
    main_args.turns = args.turns
    main_args.continuous = not args.once
    main_args.planner = resolve_planner(args)
    main_args.checkpoint_dir = str(session_dir)
    main_args.resume = str(session_dir) if checkpoint_path.exists() else None
    main_args.log_mode = args.log_mode
    return main_args


def resolve_planner(args: argparse.Namespace) -> str:
    if args.planner != "auto":
        return args.planner
    return "llm" if os.getenv("OPENROUTER_API_KEY") else "fallback"


def apply_runtime_environment(args: argparse.Namespace) -> str | None:
    if args.mode != "pyboy":
        return None
    if args.headless:
        window = "null"
    else:
        window = args.window or "SDL2"
    os.environ["PYBOY_WINDOW"] = window
    return window


def main(argv: list[str] | None = None) -> None:
    parser = create_start_parser()
    args = parser.parse_args(argv)
    window = apply_runtime_environment(args)
    main_args = build_main_args(args)
    if not args.eval:
        if main_args.resume:
            print(f"Resuming session from {main_args.resume}")
        else:
            print(f"Starting new session in {main_args.checkpoint_dir}")
        print(f"Planner mode: {main_args.planner}")
        if window:
            print(f"PyBoy window mode: {window}")
        if main_args.continuous:
            print("Watch mode is active. Press Ctrl+C to stop and keep the session checkpoint.")
    run_args(main_args)


if __name__ == "__main__":
    main()
