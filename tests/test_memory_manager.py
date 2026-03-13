from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.models.action import ActionDecision, ActionType
from pokemon_agent.models.events import EventType
from pokemon_agent.models.state import GameMode, StructuredGameState


def test_memory_manager_rebases_goals_on_current_walkthrough_milestone() -> None:
    manager = MemoryManager()
    previous = StructuredGameState(
        map_name="Pewter City",
        map_id=0x02,
        x=10,
        y=8,
        mode=GameMode.OVERWORLD,
        step=9,
        story_flags=["got_starter", "oak_received_parcel", "got_pokedex"],
    )
    current = StructuredGameState(
        map_name="Pewter City",
        map_id=0x02,
        x=10,
        y=8,
        mode=GameMode.OVERWORLD,
        step=10,
        story_flags=["got_starter", "oak_received_parcel", "got_pokedex", "beat_brock"],
        badges=["Boulder"],
    )
    manager.memory.long_term.story_flags = list(previous.story_flags)

    events = manager.update_from_transition(
        previous=previous,
        current=current,
        action=ActionDecision(action=ActionType.PRESS_A, reason="Finish Brock dialogue"),
        progress=ProgressResult("major_progress", newly_completed_subgoals=["Defeated Brock"]),
    )

    assert manager.memory.goals.long_term_goal == "Clear Mt. Moon and emerge in Cerulean City's direction."
    assert manager.memory.goals.mid_term_goal == "Go east from Pewter City across Route 3."
    assert manager.memory.long_term.story_flags == current.story_flags
    assert any(event.type == EventType.MILESTONE_COMPLETE for event in events)
    assert any("Brock" in event.summary for event in events if event.type == EventType.MILESTONE_COMPLETE)
