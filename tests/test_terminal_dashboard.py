from __future__ import annotations

from rich.console import Console

from pokemon_agent.agent.engine import TurnResult
from pokemon_agent.agent.llm_client import LLMUsage
from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.agent.validator import ActionValidator
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.progress import ProgressDetector
from pokemon_agent.agent.stuck_detector import StuckDetector
from pokemon_agent.emulator.mock import MockEmulatorAdapter
from pokemon_agent.models.action import ActionDecision, ActionType
from pokemon_agent.models.events import EventRecord, EventType
from pokemon_agent.models.state import GameMode, InventoryItem, NavigationSnapshot, PartyMember, StructuredGameState, WorldCoordinate
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
        navigation=NavigationSnapshot(
            min_x=0,
            min_y=0,
            max_x=9,
            max_y=9,
            player=WorldCoordinate(x=5, y=5),
            walkable=[WorldCoordinate(x=5, y=5), WorldCoordinate(x=5, y=4), WorldCoordinate(x=9, y=5)],
            blocked=[WorldCoordinate(x=4, y=5)],
            collision_hash="mock-hash",
        ),
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
            {
                "role": "user",
                "content": (
                    '{"context":{"candidate_next_steps":[{"id":"exit_north"}],'
                    '"overworld_context":{"visual_map":"....\\n.#..\\n..P."}}}'
                ),
            },
        ],
        prompt_metrics={"chars": 120, "approx_tokens": 30, "compact": True},
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
        "short_term_goal": "Step onto the north exit tile.",
        "mid_term_goal": "Leave Pallet Town and head toward Route 1.",
        "long_term_goal": "Reach Viridian City and continue the opening route.",
        "current_strategy": "Use the nearest safe path and confirm map transitions.",
        "pathfinding_route": ["PALLET_TOWN", "ROUTE_1", "VIRIDIAN_CITY"],
        "pathfinding_route_available": True,
        "pathfinding_target_symbol": "VIRIDIAN_CITY",
        "pathfinding_next_symbol": "ROUTE_1",
        "pathfinding_next_hop_kind": "boundary",
    }
    dashboard.status = "Running"

    console.print(dashboard.render())
    rendered = console.export_text()

    assert "Current State" in rendered
    assert "Pathfinding Route" in rendered
    assert "Turn History" in rendered
    assert "LLM Calls" in rendered
    assert "Pallet Town" in rendered
    assert "PALLET_TOWN > ROUTE_1 >" in rendered
    assert "VIRIDIAN_CITY" in rendered
    assert "MOVE_UP x1" in rendered
    assert "openai/gpt-5-mini" in rendered
    assert "Walk north to progress." in rendered
    assert "Objectives" in rendered
    assert "Step onto the north exit tile." in rendered
    assert "Leave Pallet Town and head toward Route 1." in rendered
    assert "Reach Viridian City and continue the opening route." in rendered
    assert "Planner Payload" not in rendered


def test_dashboard_renders_final_pathfinding_target_when_route_is_capped():
    console = Console(record=True, width=140, height=80)
    state = StructuredGameState(
        map_name="Pallet Town",
        map_id=0x00,
        x=5,
        y=5,
        facing="UP",
        mode=GameMode.OVERWORLD,
        navigation=NavigationSnapshot(
            min_x=0,
            min_y=0,
            max_x=9,
            max_y=9,
            player=WorldCoordinate(x=5, y=5),
            walkable=[WorldCoordinate(x=5, y=5), WorldCoordinate(x=5, y=4)],
            blocked=[WorldCoordinate(x=4, y=5)],
            collision_hash="pallet-town",
        ),
        step=20,
    )
    dashboard = TerminalDashboard(
        planner="auto_candidate",
        continuous=False,
        target_turns=4,
        checkpoint_dir="/tmp/session",
        console=console,
    )
    dashboard.current_state = state
    dashboard.summary = {
        "turns": 1,
        "fallback_turns": 0,
        "prompt_chars": 0,
        "approx_prompt_tokens": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_total_tokens": 0,
        "llm_calls": 0,
        "turns_per_call": 0.0,
        "objective_switch_rate": 0.0,
        "short_term_goal": "Cross the early-game route safely.",
        "mid_term_goal": "Reach Pewter City before continuing east.",
        "long_term_goal": "Reach Cerulean City.",
        "current_strategy": "Chunk long travel into a nearby subtarget.",
        "pathfinding_route": ["PALLET_TOWN", "ROUTE_1", "VIRIDIAN_CITY", "ROUTE_2", "PEWTER_CITY"],
        "pathfinding_route_available": True,
        "pathfinding_target_symbol": "PEWTER_CITY",
        "pathfinding_final_target_symbol": "CERULEAN_CITY",
        "pathfinding_next_symbol": "ROUTE_1",
        "pathfinding_next_hop_kind": "boundary",
    }
    dashboard.status = "Running"

    console.print(dashboard.render())
    rendered = console.export_text()

    assert "PEWTER_CITY" in rendered
    assert "CERULEAN_CITY" in rendered
    assert "Final" in rendered
    assert "Planning State" not in rendered


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
        emulator=emulator,
        memory=MemoryManager(),
        progress=ProgressDetector(),
        stuck=StuckDetector(),
        validator=ActionValidator(max_repeat=4),
        llm_client=None,
    )
