from __future__ import annotations

import argparse
import builtins
import os
import signal
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from rich import print as rich_print
except Exception:  # pragma: no cover
    rich_print = builtins.print

print = rich_print

from pokemon_agent.agent.context_manager import ContextManager
from pokemon_agent.agent.engine import ClosedLoopRunner
from pokemon_agent.agent.llm_client import OpenRouterClient
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.progress import ProgressDetector
from pokemon_agent.agent.stuck_detector import StuckDetector
from pokemon_agent.agent.validator import ActionValidator
from pokemon_agent.config import AppConfig
from pokemon_agent.emulator.mock import MockEmulatorAdapter
from pokemon_agent.emulator.pyboy_adapter import PyBoyAdapter
from pokemon_agent.ui.debug_overlay import DebugOverlayWriter
from pokemon_agent.ui.terminal_dashboard import TerminalDashboard

REPO_ROOT = Path(__file__).resolve().parents[2]


class _InterruptController:
    def __init__(self, on_first_interrupt=None) -> None:
        self.on_first_interrupt = on_first_interrupt
        self.shutdown_requested = False

    def handle_signal(self, signum, frame) -> None:  # pragma: no cover - exercised via integration
        del signum, frame
        if self.shutdown_requested:
            raise KeyboardInterrupt
        self.shutdown_requested = True
        if self.on_first_interrupt is not None:
            self.on_first_interrupt()


def build_emulator(mode: str, rom: str | None, config: AppConfig, *, live_path_overlay: bool = False):
    if mode == "pyboy":
        rom_path = rom or config.default_rom_path
        return PyBoyAdapter(rom_path, config=config, live_path_overlay=live_path_overlay)
    return MockEmulatorAdapter()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "pyboy"], default="mock")
    parser.add_argument("--rom", default=None)
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--planner", choices=["auto", "llm", "fallback"], default="fallback")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--window", choices=["SDL2", "OpenGL", "GLFW", "null"], default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--log-mode", choices=["dashboard", "verbose", "compact", "quiet"], default="dashboard")
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--debug-overlay-dir", default=None)
    parser.add_argument("--live-path-overlay", action="store_true")
    return parser


def apply_default_mode(argv: list[str] | None, default_mode: str | None = None) -> list[str] | None:
    if default_mode is None:
        return argv
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--mode" or arg.startswith("--mode=") for arg in resolved_argv):
        return resolved_argv
    return ["--mode", default_mode, *resolved_argv]


def session_is_resumable(session_dir: Path) -> bool:
    return (session_dir / "checkpoint.json").exists() and (session_dir / "emulator.state").exists()


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


def clear_pyboy_rom_sidecars(rom_path: str | Path) -> None:
    rom_file = Path(rom_path)
    for suffix in (".ram", ".sav"):
        candidate = rom_file.with_suffix(rom_file.suffix + suffix)
        if candidate.exists():
            candidate.unlink()


def build_main_args(args: argparse.Namespace) -> argparse.Namespace:
    debug_overlay_path = getattr(args, "debug_overlay_dir", None)
    debug_overlay_enabled = bool(getattr(args, "debug_overlay", False) or debug_overlay_path)
    session_dir = None if not args.session_dir else (REPO_ROOT / args.session_dir).resolve()
    if args.fresh and session_dir and session_dir.exists():
        shutil.rmtree(session_dir)
    if session_dir is not None:
        session_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    rom = args.rom
    if mode == "pyboy":
        resolved_rom = str((REPO_ROOT / (rom or AppConfig().default_rom_path)).resolve())
        if args.fresh:
            clear_pyboy_rom_sidecars(resolved_rom)
        if rom is not None:
            rom = resolved_rom
    continuous = args.continuous or (session_dir is not None and not args.once)
    checkpoint_dir = str(session_dir) if session_dir is not None else args.checkpoint_dir
    debug_overlay_dir = None
    if debug_overlay_enabled:
        if debug_overlay_path:
            debug_overlay_dir = str((REPO_ROOT / debug_overlay_path).resolve())
        elif session_dir is not None:
            debug_overlay_dir = str((session_dir / "debug_overlay").resolve())
        elif checkpoint_dir is not None:
            debug_overlay_dir = str((Path(checkpoint_dir) / "debug_overlay").resolve())
        else:
            debug_overlay_dir = str((REPO_ROOT / ".debug_overlay").resolve())
    resume = args.resume
    if session_dir is not None:
        resume = str(session_dir) if session_is_resumable(session_dir) else None

    return argparse.Namespace(
        mode=mode,
        rom=rom,
        turns=args.turns,
        continuous=continuous,
        planner=resolve_planner(args),
        checkpoint_dir=checkpoint_dir,
        debug_overlay_dir=debug_overlay_dir,
        resume=resume,
        log_mode=args.log_mode,
        live_path_overlay=bool(getattr(args, "live_path_overlay", False)),
    )


def _print_turn(result, log_mode: str) -> None:
    if log_mode == "quiet":
        return
    if log_mode == "compact":
        summary: dict[str, Any] = {
            "turn": result.turn_index,
            "action": result.action.action.value,
            "repeat": result.action.repeat,
            "classification": result.progress.classification,
            "map": result.after.map_name,
            "position": {"x": result.after.x, "y": result.after.y},
            "mode": result.after.mode.value,
            "planner_source": result.planner_source,
            "stuck": result.stuck_state.score,
            "fallback": result.used_fallback,
            "llm_attempted": result.llm_attempted,
        }
        if result.llm_usage and result.llm_usage.total_tokens is not None:
            summary["llm_tokens"] = result.llm_usage.total_tokens
        if result.events:
            summary["events"] = [event.summary for event in result.events]
        if result.stuck_state.recovery_hint:
            summary["recovery_hint"] = result.stuck_state.recovery_hint
        if result.llm_attempted and result.used_fallback and result.raw_model_response:
            summary["llm_error"] = result.raw_model_response
        print(summary)
        return

    print(
        f"[bold]Turn {result.turn_index}[/bold] "
        f"action={result.action.action.value} repeat={result.action.repeat} "
        f"classification={result.progress.classification} "
        f"source={result.planner_source} "
        f"stuck={result.stuck_state.score} fallback={result.used_fallback}"
    )
    if result.prompt_metrics:
        print(
            "  prompt",
            {
                "chars": result.prompt_metrics.get("chars"),
                "approx_tokens": result.prompt_metrics.get("approx_tokens"),
                "warning": result.prompt_metrics.get("warning"),
            },
        )
    print("  before", result.before.prompt_summary())
    print("  after ", result.after.prompt_summary())
    if result.events:
        print("  events", [event.model_dump() for event in result.events])
    if result.stuck_state.recovery_hint:
        print("  recovery_hint", result.stuck_state.recovery_hint)


def run_args(args: argparse.Namespace) -> None:
    config = AppConfig()
    emulator = build_emulator(
        args.mode,
        args.rom,
        config,
        live_path_overlay=bool(getattr(args, "live_path_overlay", False)),
    )
    memory = MemoryManager(window=config.short_term_window)
    progress = ProgressDetector()
    stuck = StuckDetector(threshold=config.stuck_threshold)
    context_manager = ContextManager(
        budget_tokens=config.prompt_budget_tokens,
        action_window=config.context_action_window,
        event_window=config.context_event_window,
    )
    validator = ActionValidator(max_repeat=config.max_repeat)
    if args.planner == "llm" and not config.openrouter.api_key:
        raise SystemExit("Planner is set to 'llm' but OPENROUTER_API_KEY is missing.")
    llm_client = OpenRouterClient(config.openrouter) if args.planner == "llm" else None
    runner = ClosedLoopRunner(
        emulator=emulator,
        memory=memory,
        progress=progress,
        stuck=stuck,
        validator=validator,
        llm_client=llm_client,
        context_manager=context_manager,
    )

    turns = args.turns or config.max_turns
    dashboard = None
    overlay_writer = DebugOverlayWriter(args.debug_overlay_dir) if args.debug_overlay_dir else None
    if args.log_mode == "dashboard":
        dashboard = TerminalDashboard(
            planner=args.planner,
            continuous=args.continuous,
            target_turns=turns,
            checkpoint_dir=args.checkpoint_dir,
        )
    def _request_stop() -> None:
        if dashboard is not None:
            dashboard.request_stop()
            return
        if args.log_mode != "quiet":
            print("[bold]Stop requested[/bold] Press Ctrl+C again to save immediately and exit.")
    interrupt_controller = _InterruptController(on_first_interrupt=_request_stop)
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, interrupt_controller.handle_signal)

    interrupted = False
    shutdown_checkpoint_saved = False
    shutdown_message_printed = False
    try:
        if args.resume:
            payload = runner.load_checkpoint(args.resume)
            if dashboard is not None:
                dashboard.record_resume(args.resume, int(payload.get("completed_turns", 0)))
            elif args.log_mode != "quiet":
                print("[bold]Resumed checkpoint[/bold]", payload.get("completed_turns", 0))
        if args.log_mode not in {"quiet", "dashboard"}:
            print("[bold]Planner[/bold]", args.planner)
        initial = emulator.get_structured_state()
        if dashboard is not None:
            dashboard.start(initial, runner.summary())
        elif args.log_mode != "quiet":
            print("[bold]Initial state[/bold]", initial.model_dump())
        start_turn = runner.completed_turns + 1
        turn_index = start_turn
        while args.continuous or turn_index < start_turn + turns:
            if interrupt_controller.shutdown_requested:
                interrupted = True
                break
            result = runner.run_turn(turn_index)
            if dashboard is not None:
                dashboard.update_turn(result, runner.summary())
            else:
                _print_turn(result, args.log_mode)
            if overlay_writer is not None:
                overlay_writer.write_turn(result, runner.summary())
            if args.checkpoint_dir:
                runner.save_checkpoint(args.checkpoint_dir)
            turn_index += 1
            if interrupt_controller.shutdown_requested:
                interrupted = True
                break
    except KeyboardInterrupt:
        interrupted = True
        if dashboard is None and args.log_mode != "quiet":
            print("[bold]Stopped[/bold] Saving checkpoint and closing emulator.")
            shutdown_message_printed = True
        if args.checkpoint_dir:
            runner.save_checkpoint(args.checkpoint_dir)
            shutdown_checkpoint_saved = True
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        if interrupt_controller.shutdown_requested and not interrupted:
            interrupted = True
        if interrupt_controller.shutdown_requested and dashboard is None and args.log_mode != "quiet" and not shutdown_message_printed:
            print("[bold]Stopped[/bold] Saving checkpoint and closing emulator.")
        if interrupt_controller.shutdown_requested and args.checkpoint_dir and not shutdown_checkpoint_saved:
            runner.save_checkpoint(args.checkpoint_dir)
        if dashboard is not None:
            dashboard.finish(interrupted=interrupted)
        elif args.log_mode != "quiet":
            print("[bold]Run summary[/bold]", runner.summary())
        emulator.close()


def main(argv: list[str] | None = None, default_mode: str | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(apply_default_mode(argv, default_mode))
    window = apply_runtime_environment(args)
    main_args = build_main_args(args)
    if args.session_dir:
        if main_args.resume:
            print(f"Resuming session from {main_args.resume}")
        else:
            print(f"Starting new session in {main_args.checkpoint_dir}")
        print(f"Planner mode: {main_args.planner}")
        if window:
            print(f"PyBoy window mode: {window}")
        if main_args.continuous:
            print("Watch mode is active. Press Ctrl+C once to stop after the current turn, or twice to save and exit immediately.")
        if main_args.debug_overlay_dir:
            print(f"Debug overlay output: {main_args.debug_overlay_dir}")
    run_args(main_args)


if __name__ == "__main__":
    main()
