from pokemon_agent.agent.context_manager import ContextManager
from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.events import EventRecord
from pokemon_agent.models.events import EventType
from pokemon_agent.models.memory import MemoryState
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState


def test_context_manager_omits_stuck_warning_section() -> None:
    manager = ContextManager()
    state = StructuredGameState(
        map_name="Mock Town",
        map_id="mock_town",
        x=5,
        y=5,
        mode=GameMode.OVERWORLD,
        step=12,
    )
    manager.record_turn(
        turn_index=1,
        action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="check interaction"),
        after_state=state,
        progress=ProgressResult("no_effect"),
        events=[EventRecord(type=EventType.NO_EFFECT, summary="PRESS_A had no effect", step=12)],
        stuck_state=StuckState(score=4, steps_since_progress=4, oscillating=True),
        used_fallback=False,
        llm_attempted=False,
        planner_source="recovery_controller",
    )

    snapshot = manager.build_snapshot(
        state,
        MemoryState(),
        stuck_state=StuckState(score=6, steps_since_progress=6, oscillating=True),
    )

    assert "stuck_warning" not in snapshot.payload["context"]
    assert snapshot.payload["context"]["last_candidate_result"]["stuck_score"] == 4
