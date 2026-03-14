import json

from pokemon_agent.agent.context_manager import ContextManager
from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.models.action import ActionDecision, ActionType
from pokemon_agent.models.events import EventRecord, EventType
from pokemon_agent.models.memory import MemoryState
from pokemon_agent.models.planner import CandidateNextStep, Objective, ObjectiveHorizon, ObjectiveTarget
from pokemon_agent.models.state import GameMode, NavigationSnapshot, PartyMember, StructuredGameState, WorldCoordinate


def _screen_grids() -> tuple[list[list[int]], list[list[int]]]:
    game_area = [[0 for _ in range(20)] for _ in range(18)]
    collision_area = [[0 for _ in range(20)] for _ in range(18)]

    for row in (4, 5):
        for col in (4, 5):
            game_area[row][col] = 99
            collision_area[row][col] = 1

    collision_area[8][6] = 1
    collision_area[8][8] = 1
    collision_area[9][0] = 1
    collision_area[10][14] = 1
    collision_area[10][15] = 1
    return game_area, collision_area


def _memory_state() -> MemoryState:
    memory_state = MemoryState()
    memory_state.goals.active_objectives = [
        Objective(
            id="long",
            horizon=ObjectiveHorizon.LONG_TERM,
        ),
        Objective(
            id="mid",
            horizon=ObjectiveHorizon.MID_TERM,
            target=ObjectiveTarget(kind="exit", map_name="Viridian City"),
        ),
        Objective(
            id="short",
            horizon=ObjectiveHorizon.SHORT_TERM,
            target=ObjectiveTarget(kind="map", map_name="Viridian City", x=7, y=8),
        ),
    ]
    return memory_state


def _state() -> StructuredGameState:
    game_area, collision_area = _screen_grids()
    return StructuredGameState(
        map_name="Viridian City",
        map_id=0x01,
        x=7,
        y=8,
        facing="LEFT",
        mode=GameMode.OVERWORLD,
        story_flags=["got_starter", "oak_received_parcel", "got_pokedex"],
        navigation=NavigationSnapshot(
            min_x=0,
            min_y=0,
            max_x=9,
            max_y=9,
            player=WorldCoordinate(x=7, y=8),
            walkable=[
                WorldCoordinate(x=7, y=8),
                WorldCoordinate(x=7, y=7),
                WorldCoordinate(x=9, y=8),
                WorldCoordinate(x=0, y=8),
            ],
            blocked=[WorldCoordinate(x=6, y=8)],
            collision_hash="collision-hash",
        ),
        step=42,
        game_area=game_area,
        collision_area=collision_area,
    )


def _candidates() -> list[CandidateNextStep]:
    return [
        CandidateNextStep(
            id="exit_east",
            type="GO_TO_MAP_EXIT",
            target=ObjectiveTarget(kind="exit", map_name="Viridian City", x=9, y=8, detail="east"),
            why="The east edge is reachable in 2 steps and may reveal a transition.",
            priority=60,
            expected_success_signal="Map changes or new options appear",
        ),
        CandidateNextStep(
            id="frontier_west",
            type="EXPLORE_NEAREST_FRONTIER",
            target=ObjectiveTarget(kind="frontier", map_name="Viridian City", x=0, y=8),
            why="The west boundary is the nearest frontier tile.",
            priority=42,
            expected_success_signal="Position changes or new candidates appear",
        ),
    ]


def test_context_manager_builds_tight_overworld_payload():
    snapshot = ContextManager().build_snapshot(
        _state(),
        _memory_state(),
        candidate_next_steps=_candidates(),
    )

    context = snapshot.payload["context"]
    text = json.dumps(snapshot.payload, sort_keys=True)

    assert context["immediate_state"]["map"]["name"] == "Viridian City"
    assert context["immediate_state"]["current_milestone"]["id"] == "viridian_forest"
    assert "story_progress" not in context["immediate_state"]
    assert "navigation_window" not in context["immediate_state"]
    assert "walkthrough_context" not in context
    assert "active_objective_stack" not in context
    assert "recommended_next_step" not in context
    assert "goal" in context
    assert "Viridian Forest" in context["goal"]
    assert context["overworld_context"]["visual_map"] is not None
    assert "P" in context["overworld_context"]["visual_map"]
    assert "~" in context["overworld_context"]["visual_map"]
    assert "@" in context["overworld_context"]["visual_map"]
    assert "D" in context["overworld_context"]["visual_map"]
    assert context["candidate_next_steps"][0]["id"] == "exit_east"
    assert "candidate_next_steps" in snapshot.system_prompt
    assert '"walkable":' not in text
    assert '"blocked":' not in text
    assert "party" not in text
    assert "inventory" not in text




def test_context_manager_keeps_deterministic_text_context_small():
    state = StructuredGameState(
        map_name="Oak's Lab",
        map_id=0x28,
        x=4,
        y=6,
        mode=GameMode.TEXT,
        text_box_open=True,
        metadata={"dialogue_text": "Oak: Take this?", "yes_no_prompt": True},
        step=15,
    )

    snapshot = ContextManager().build_snapshot(state, _memory_state())
    context = snapshot.payload["context"]

    assert context["immediate_state"]["ui_flags"]["text_box_open"] is True
    assert context["immediate_state"]["ui_flags"]["yes_no_prompt"] is True
    assert context["immediate_state"]["dialogue_text"] == "Oak: Take this?"
    assert context["immediate_state"]["current_milestone"]["id"] == "get_starter"
    assert context["dialogue_context"]["dialogue_text"] == "Oak: Take this?"
    assert context["dialogue_context"]["choice_mode"] == "ADVANCE_OR_CHOOSE"
    assert "select_yes or select_no" in snapshot.system_prompt
    assert "candidate_next_steps" not in context
    assert "recommended_next_step" not in context


def test_context_manager_marks_yes_no_dialogue_prompts():
    state = StructuredGameState(
        map_name="Oak's Lab",
        map_id=0x28,
        x=4,
        y=6,
        mode=GameMode.TEXT,
        text_box_open=True,
        metadata={"dialogue": "Professor Oak: Do you want a Pokedex? YES/NO"},
        step=15,
    )

    snapshot = ContextManager().build_snapshot(state, _memory_state())
    context = snapshot.payload["context"]

    assert context["dialogue_context"]["choice_mode"] == "YES_NO"
    assert context["dialogue_context"]["dialogue_text"] == "Professor Oak: Do you want a Pokedex? YES/NO"
    assert "select_yes or select_no" in snapshot.system_prompt


def test_context_manager_builds_battle_specific_prompt_and_context():
    state = StructuredGameState(
        map_name="Route 1",
        map_id=0x02,
        x=5,
        y=2,
        mode=GameMode.BATTLE,
        battle_state={
            "kind": "WILD",
            "opponent": "PIDGEY",
            "opponent_level": 3,
            "moves": ["Tackle", "Growl"],
        },
        party=[PartyMember(name="Charmander", level=8, hp=18, max_hp=20)],
        step=22,
    )

    snapshot = ContextManager().build_snapshot(state, _memory_state())
    context = snapshot.payload["context"]

    assert context["battle_context"]["enemy"]["name"] == "PIDGEY"
    assert context["battle_context"]["enemy"]["level"] == 3
    assert context["battle_context"]["lead_pokemon"]["name"] == "Charmander"
    assert context["battle_context"]["moves"] == ["Tackle", "Growl"]
    assert "battle_prompt" not in context["battle_context"]
    assert "candidate_next_steps" in snapshot.system_prompt


def test_context_manager_includes_stuck_warning_and_last_outcome():
    manager = ContextManager()
    state = _state()
    manager.record_turn(
        turn_index=3,
        action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="Check nearby interaction"),
        after_state=state,
        progress=ProgressResult("interaction_success"),
        events=[EventRecord(type=EventType.TEXT_OPENED, summary="Opened dialogue/text box", step=42)],
        stuck_state=StuckState(score=3, recovery_hint="Try a new local interaction"),
        used_fallback=False,
        llm_attempted=False,
        planner_source="auto_candidate",
    )

    snapshot = manager.build_snapshot(
        state,
        _memory_state(),
        stuck_state=StuckState(score=4, recovery_hint="Try a new local interaction"),
        candidate_next_steps=_candidates(),
    )

    context = snapshot.payload["context"]
    assert context["last_outcome"]["action"] == "PRESS_A"
    assert context["stuck_warning"]["stuck_score"] == 4


def test_context_manager_prunes_recent_events_before_objectives_and_candidates():
    manager = ContextManager(budget_tokens=1)
    memory_state = _memory_state()
    memory_state.recent_events = [
        EventRecord(type=EventType.MOVED, summary=f"Moved {index}", step=index) for index in range(4)
    ]

    snapshot = manager.build_snapshot(
        _state(),
        memory_state,
        stuck_state=StuckState(score=4, recovery_hint="Try a new local interaction"),
        candidate_next_steps=_candidates(),
    )

    assert "recent_events" in snapshot.dropped_sections
    assert "candidate_next_steps" in snapshot.payload["context"]
    assert "goal" in snapshot.payload["context"]
