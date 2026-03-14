from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.models.action import ActionDecision, ActionType
from pokemon_agent.models.events import EventType
from pokemon_agent.models.state import GameMode, NavigationSnapshot, StructuredGameState, WorldCoordinate


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


def test_memory_manager_discovers_tiles_and_confirms_connectors() -> None:
    manager = MemoryManager()
    previous = StructuredGameState(
        map_name="Mock Town",
        map_id="mock_town",
        x=8,
        y=5,
        mode=GameMode.OVERWORLD,
        step=11,
        navigation=NavigationSnapshot(
            min_x=0,
            min_y=0,
            max_x=9,
            max_y=9,
            player=WorldCoordinate(x=8, y=5),
            walkable=[WorldCoordinate(x=8, y=5), WorldCoordinate(x=8, y=4), WorldCoordinate(x=9, y=4)],
            blocked=[WorldCoordinate(x=9, y=5)],
            collision_hash="mock-town",
        ),
    )
    current = StructuredGameState(
        map_name="Route 1",
        map_id="route_1",
        x=1,
        y=5,
        mode=GameMode.OVERWORLD,
        step=12,
        navigation=NavigationSnapshot(
            min_x=0,
            min_y=0,
            max_x=11,
            max_y=7,
            player=WorldCoordinate(x=1, y=5),
            walkable=[WorldCoordinate(x=1, y=5), WorldCoordinate(x=0, y=5), WorldCoordinate(x=2, y=5)],
            blocked=[],
            collision_hash="route-1",
        ),
    )

    events = manager.update_from_transition(
        previous=previous,
        current=current,
        action=ActionDecision(action=ActionType.MOVE_RIGHT, reason="Step into the discovered warp"),
        progress=ProgressResult("major_progress", ["map_name"], ["Entered Route 1"]),
    )

    world_map = manager.memory.long_term.world_map
    assert "Mock Town" in world_map.maps
    assert "Route 1" in world_map.maps
    connector = world_map.connectors["Mock Town::side::east"]
    assert connector.status.value == "confirmed"
    assert connector.destination_map == "Route 1"
    assert connector.approach_x == 8
    assert connector.approach_y == 5
    assert connector.transition_action == ActionType.MOVE_RIGHT
    reverse = world_map.connectors["Route 1::side::west"]
    assert reverse.destination_map == "Mock Town"
    assert any(event.type == EventType.CONNECTOR_CONFIRMED for event in events)
