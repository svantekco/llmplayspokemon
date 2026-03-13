from __future__ import annotations

from rich.console import Console

from pokemon_agent.agent.engine import TurnResult
from pokemon_agent.agent.llm_client import LLMUsage
from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.agent.prompt_builder import PromptMetrics
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.agent.validator import ActionValidator
from pokemon_agent.agent.executor import Executor
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.progress import ProgressDetector
from pokemon_agent.agent.prompt_builder import PromptBuilder
from pokemon_agent.agent.stuck_detector import StuckDetector
from pokemon_agent.emulator.mock import MockEmulatorAdapter
from pokemon_agent.models.action import ActionDecision, ActionType
from pokemon_agent.models.events import EventRecord, EventType
from pokemon_agent.models.state import GameMode, InventoryItem, PartyMember, StructuredGameState
from pokemon_agent.ui.terminal_dashboard import TerminalDashboard


def test_dashboard_renders_state_turns_and_llm_details():
    console = Console(record=True, width=140, height=80)
    before = StructuredGameState(
        map_name="Pallet Town",
        map_id=1,
        x=5,
        y=5,
        facing="UP",
        mode=GameMode.OVERWORLD,
        party=[PartyMember(name="Charmander", hp=20, max_hp=20)],
        inventory=[InventoryItem(name="Potion", count=1)],
        step=12,
    )
    after = before.model_copy(update={"y": 4, "step": 13})
    turn = TurnResult(
        turn_index=1,
        before=before,
        action=ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="Head north"),
        after=after,
        progress=ProgressResult("movement_success", ["position"], [], ["Position changed"]),
        stuck_state=StuckState(score=0),
        events=[EventRecord(type=EventType.MOVED, summary="Moved to (5, 4)", step=13)],
        used_fallback=False,
        raw_model_response='{"action":"MOVE_UP","repeat":1,"reason":"Walk north to progress."}',
        prompt_messages=[
            {"role": "system", "content": "You are a planner."},
            {"role": "user", "content": '{"context":{"candidate_next_steps":[{"id":"exit_north"}]}}'},
        ],
        prompt_metrics=PromptMetrics(chars=120, approx_tokens=30, compact=True),
        llm_usage=LLMUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        llm_attempted=True,
        llm_model="openai/gpt-5-mini",
    )
    dashboard = TerminalDashboard(
        planner="llm",
        continuous=False,
        target_turns=4,
        checkpoint_dir="/tmp/session",
        console=console,
    )
    dashboard.record_resume("/tmp/session", completed_turns=2)
    dashboard.current_state = after
    dashboard.latest_turn = turn
    dashboard.turn_history = [turn]
    dashboard.summary = {
        "turns": 3,
        "fallback_turns": 0,
        "prompt_chars": 120,
        "approx_prompt_tokens": 30,
        "llm_prompt_tokens": 20,
        "llm_completion_tokens": 10,
        "llm_total_tokens": 30,
        "llm_calls": 1,
        "turns_per_call": 3.0,
        "objective_switch_rate": 0.0,
    }
    dashboard.status = "Running"

    console.print(dashboard.render())
    rendered = console.export_text()

    assert "Current State" in rendered
    assert "Turn History" in rendered
    assert "LLM Calls" in rendered
    assert "Pallet Town" in rendered
    assert "MOVE_UP x1" in rendered
    assert "openai/gpt-5-mini" in rendered
    assert "Walk north to progress." in rendered


def test_dashboard_shows_fallback_note_when_no_llm_call():
    console = Console(record=True, width=140, height=80)
    emulator = MockEmulatorAdapter()
    runner = _build_runner(emulator)
    dashboard = TerminalDashboard(
        planner="fallback",
        continuous=False,
        target_turns=1,
        console=console,
    )

    dashboard.start(emulator.get_structured_state(), runner.summary())
    result = runner.run_turn(1)
    dashboard.update_turn(result, runner.summary())
    dashboard.finish()

    rendered = console.export_text()

    assert "No network call was made" in rendered
    assert "auto" in rendered or "skip" in rendered


def _build_runner(emulator: MockEmulatorAdapter):
    from pokemon_agent.agent.engine import ClosedLoopRunner

    return ClosedLoopRunner(
        executor=Executor(emulator),
        memory=MemoryManager(),
        progress=ProgressDetector(),
        stuck=StuckDetector(),
        prompts=PromptBuilder(),
        validator=ActionValidator(max_repeat=4),
        llm_client=None,
    )
