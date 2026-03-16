import json

from pokemon_agent.agent.context_manager import ContextManager, build_messages, measure_prompt
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.emulator.screen_renderer import build_ascii_map
from pokemon_agent.models.planner import CandidateNextStep
from pokemon_agent.models.planner import Objective
from pokemon_agent.models.planner import ObjectiveHorizon
from pokemon_agent.models.planner import ObjectiveTarget
from pokemon_agent.models.planner import StrategicObjective
from pokemon_agent.models.state import GameMode, NavigationSnapshot, StructuredGameState, WorldCoordinate


def test_prompt_builder_returns_two_messages():
    memory = MemoryManager()
    memory.memory.goals.active_objectives = [
        Objective(id="long", horizon=ObjectiveHorizon.LONG_TERM),
        Objective(id="mid", horizon=ObjectiveHorizon.MID_TERM),
        Objective(id="short", horizon=ObjectiveHorizon.SHORT_TERM),
    ]
    context_manager = ContextManager()
    snapshot = context_manager.build_snapshot(StructuredGameState(), memory.memory)
    messages = build_messages(snapshot)
    assert len(messages) == 2
    payload = json.loads(messages[1]["content"])
    assert "context" in payload
    assert "response_schema" in payload
    assert "allowed_actions" not in payload
    metrics = measure_prompt(messages, snapshot)
    assert metrics["chars"] > 0
    assert metrics["approx_tokens"] > 0
    assert metrics["budget_tokens"] == snapshot.budget_tokens
    assert metrics["used_tokens"] == snapshot.used_tokens
    assert isinstance(metrics["section_tokens"], dict)


def test_prompt_builder_uses_candidate_based_context():
    memory = MemoryManager()
    memory.memory.goals.active_objectives = [
        Objective(id="long", horizon=ObjectiveHorizon.LONG_TERM),
        Objective(id="mid", horizon=ObjectiveHorizon.MID_TERM),
        Objective(id="short", horizon=ObjectiveHorizon.SHORT_TERM),
    ]
    context_manager = ContextManager()
    state = StructuredGameState(
        map_name="Mock Town",
        x=5,
        y=5,
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
    )
    candidates = [
        CandidateNextStep(
            id="exit_east",
            type="GO_TO_MAP_EXIT",
            target=ObjectiveTarget(kind="exit", map_name="Mock Town", x=9, y=5),
            why="The east edge is reachable and may reveal a transition.",
            priority=60,
            expected_success_signal="Map changes or new options appear",
        )
    ]

    snapshot = context_manager.build_snapshot(state, memory.memory, candidate_next_steps=candidates)
    messages = build_messages(snapshot)
    payload = json.loads(messages[1]["content"])
    text = messages[1]["content"]

    assert "candidate_id" in messages[0]["content"]
    assert "candidate_next_steps" in messages[0]["content"]
    assert "candidate_next_steps" in payload["context"]
    assert "overworld_context" in payload["context"]
    assert payload["context"]["overworld_context"]["visual_map"] is not None
    assert payload["context"]["candidate_next_steps"][0]["type"] == "GO_TO_MAP_EXIT"
    assert '"walkable":' not in text
    assert '"blocked":' not in text


def test_prompt_builder_serializes_current_game_area_ascii_map():
    memory = MemoryManager()
    context_manager = ContextManager()
    game_area = [[0 for _ in range(20)] for _ in range(18)]
    collision_area = [[1 for _ in range(20)] for _ in range(18)]
    for row in (3, 4):
        for col in (3, 4):
            game_area[row][col] = 99
            collision_area[row][col] = 0
    collision_area[7][0] = 0

    state = StructuredGameState(
        map_name="Viridian City",
        map_id=0x01,
        x=7,
        y=8,
        mode=GameMode.OVERWORLD,
        navigation=NavigationSnapshot(
            min_x=0,
            min_y=0,
            max_x=9,
            max_y=9,
            player=WorldCoordinate(x=7, y=8),
            walkable=[WorldCoordinate(x=7, y=8)],
            blocked=[WorldCoordinate(x=7, y=0)],
            collision_hash="mock-hash",
        ),
        game_area=game_area,
        collision_area=collision_area,
    )

    snapshot = context_manager.build_snapshot(state, memory.memory)
    messages = build_messages(snapshot)
    payload = json.loads(messages[1]["content"])

    assert payload["context"]["overworld_context"]["visual_map"] == build_ascii_map(state)


def test_prompt_builder_builds_objective_planner_snapshot():
    memory = MemoryManager()
    memory.memory.long_term.objective = StrategicObjective(
        goal="Move toward Oak's Lab",
        target_map="Oak's Lab",
        strategy="Keep the plan symbolic.",
        milestone_id="get_starter",
        confidence=0.9,
        generated_at_step=12,
        generated_at_map="Red's House 2F",
    )
    context_manager = ContextManager()
    state = StructuredGameState(map_name="Red's House 2F", map_id=0x25, x=3, y=3, mode=GameMode.OVERWORLD)

    snapshot = context_manager.build_objective_snapshot(state, memory.memory, replan_reason="missing_plan")
    payload = json.loads(build_messages(snapshot)[1]["content"])

    assert "goal" in payload["response_schema"]
    assert payload["context"]["planner_kind"] == "objective"
    assert payload["context"]["replan_reason"] == "missing_plan"
    assert payload["context"]["current_objective"]["target_map"] == "Oak's Lab"
