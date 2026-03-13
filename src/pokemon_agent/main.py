from __future__ import annotations

import argparse
import builtins
from typing import Any

try:
    from rich import print
except Exception:  # pragma: no cover
    print = builtins.print

from pokemon_agent.agent.evaluator import ScenarioEvaluator
from pokemon_agent.agent.context_manager import ContextManager
from pokemon_agent.agent.engine import ClosedLoopRunner
from pokemon_agent.agent.executor import Executor
from pokemon_agent.agent.llm_client import OpenRouterClient
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.progress import ProgressDetector
from pokemon_agent.agent.prompt_builder import PromptBuilder
from pokemon_agent.agent.stuck_detector import StuckDetector
from pokemon_agent.agent.validator import ActionValidator
from pokemon_agent.config import AppConfig
from pokemon_agent.emulator.mock import MockEmulatorAdapter
from pokemon_agent.emulator.pyboy_adapter import PyBoyAdapter
from pokemon_agent.ui.terminal_dashboard import TerminalDashboard


def build_emulator(mode: str, rom: str | None, config: AppConfig):
    if mode == "pyboy":
        rom_path = rom or config.default_rom_path
        return PyBoyAdapter(rom_path, config=config)
    return MockEmulatorAdapter()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "pyboy"], default="mock")
    parser.add_argument("--rom", default=None)
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--planner", choices=["llm", "fallback"], default="fallback")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log-mode", choices=["dashboard", "verbose", "compact", "quiet"], default="dashboard")
    parser.add_argument("--eval", action="store_true")
    return parser


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
                "chars": result.prompt_metrics.chars,
                "approx_tokens": result.prompt_metrics.approx_tokens,
                "warning": result.prompt_metrics.warning,
            },
        )
    print("  before", result.before.prompt_summary())
    print("  after ", result.after.prompt_summary())
    if result.events:
        print("  events", [event.model_dump() for event in result.events])
    if result.stuck_state.recovery_hint:
        print("  recovery_hint", result.stuck_state.recovery_hint)


def run_args(args: argparse.Namespace) -> None:
    if args.eval:
        results = ScenarioEvaluator().run()
        for result in results:
            print(
                f"[bold]{result.name}[/bold] passed={result.passed} "
                f"turns={result.turns} move={result.movement_success} "
                f"interaction={result.interaction_success} major={result.major_progress} "
                f"max_stuck={result.max_stuck}"
            )
            if result.notes:
                print("  notes", result.notes)
        return

    config = AppConfig()
    emulator = build_emulator(args.mode, args.rom, config)
    executor = Executor(emulator)
    memory = MemoryManager(window=config.short_term_window)
    progress = ProgressDetector()
    stuck = StuckDetector(threshold=config.stuck_threshold)
    context_manager = ContextManager(
        budget_tokens=config.prompt_budget_tokens,
        action_window=config.context_action_window,
        event_window=config.context_event_window,
    )
    prompts = PromptBuilder()
    validator = ActionValidator(max_repeat=config.max_repeat)
    if args.planner == "llm" and not config.openrouter.api_key:
        raise SystemExit("Planner is set to 'llm' but OPENROUTER_API_KEY is missing.")
    llm_client = OpenRouterClient(config.openrouter) if args.planner == "llm" else None
    runner = ClosedLoopRunner(
        executor=executor,
        memory=memory,
        progress=progress,
        stuck=stuck,
        prompts=prompts,
        validator=validator,
        llm_client=llm_client,
        context_manager=context_manager,
    )

    turns = args.turns or config.max_turns
    dashboard = None
    if args.log_mode == "dashboard":
        dashboard = TerminalDashboard(
            planner=args.planner,
            continuous=args.continuous,
            target_turns=turns,
            checkpoint_dir=args.checkpoint_dir,
        )

    interrupted = False
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
            result = runner.run_turn(turn_index)
            if dashboard is not None:
                dashboard.update_turn(result, runner.summary())
            else:
                _print_turn(result, args.log_mode)
            if args.checkpoint_dir:
                runner.save_checkpoint(args.checkpoint_dir)
            turn_index += 1
    except KeyboardInterrupt:
        interrupted = True
        if dashboard is None and args.log_mode != "quiet":
            print("[bold]Stopped[/bold] Saving checkpoint and closing emulator.")
        if args.checkpoint_dir:
            runner.save_checkpoint(args.checkpoint_dir)
    finally:
        if dashboard is not None:
            dashboard.finish(interrupted=interrupted)
        elif args.log_mode != "quiet":
            print("[bold]Run summary[/bold]", runner.summary())
        emulator.close()


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    run_args(args)


if __name__ == "__main__":
    main()
