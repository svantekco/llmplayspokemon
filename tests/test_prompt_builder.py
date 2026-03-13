import json

from pokemon_agent.agent.context_manager import ContextManager
from pokemon_agent.agent.memory_manager import MemoryManager
from pokemon_agent.agent.prompt_builder import PromptBuilder
from pokemon_agent.models.planner import CandidateNextStep, Objective, ObjectiveHorizon, ObjectiveTarget
from pokemon_agent.models.state import NavigationSnapshot, StructuredGameState, WorldCoordinate


def test_prompt_builder_returns_two_messages():
    memory = MemoryManager()
    memory.memory.goals.active_objectives = [
        Objective(id="long", horizon=ObjectiveHorizon.LONG_TERM, summary="Advance the main story", priority=30),
        Objective(id="mid", horizon=ObjectiveHorizon.MID_TERM, summary="Find the next exit", priority=20),
        Objective(id="short", horizon=ObjectiveHorizon.SHORT_TERM, summary="Take the best local step", priority=10),
    ]
    context_manager = ContextManager()
    builder = PromptBuilder()
    snapshot = context_manager.build_snapshot(StructuredGameState(), memory.memory)
    messages = builder.build(snapshot)
    assert len(messages) == 2
    payload = json.loads(messages[1]["content"])
    assert "context" in payload
    assert "response_schema" in payload
    assert "allowed_actions" not in payload
    metrics = builder.measure(messages, snapshot)
    assert metrics.chars > 0
    assert metrics.approx_tokens > 0
    assert metrics.budget_tokens == snapshot.budget_tokens
    assert metrics.used_tokens == snapshot.used_tokens
    assert isinstance(metrics.section_tokens, dict)


def test_prompt_builder_uses_candidate_based_context():
    memory = MemoryManager()
    memory.memory.goals.active_objectives = [
        Objective(id="long", horizon=ObjectiveHorizon.LONG_TERM, summary="Advance the main story", priority=30),
        Objective(id="mid", horizon=ObjectiveHorizon.MID_TERM, summary="Exit the current map", priority=20),
        Objective(id="short", horizon=ObjectiveHorizon.SHORT_TERM, summary="Choose the best exit", priority=10),
    ]
    context_manager = ContextManager()
    builder = PromptBuilder()
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

    snapshot = context_manager.build_snapshot(state, memory.memory, candidate_next_steps=candidates, recommended_step=candidates[0])
    messages = builder.build(snapshot)
    payload = json.loads(messages[1]["content"])
    text = messages[1]["content"]

    assert "candidate_id" in messages[0]["content"]
    assert "Overworld:" in messages[0]["content"]
    assert "candidate_next_steps" in payload["context"]
    assert "overworld_context" in payload["context"]
    assert payload["context"]["overworld_context"]["visual_map"] is not None
    assert payload["context"]["candidate_next_steps"][0]["type"] == "GO_TO_MAP_EXIT"
    assert '"walkable":' not in text
    assert '"blocked":' not in text
