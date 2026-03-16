import json

from pokemon_agent.agent.context_manager import ContextManager
from pokemon_agent.agent.objective_manager import ObjectiveManager
from pokemon_agent.models.memory import MemoryState
from pokemon_agent.models.planner import StrategicObjective
from pokemon_agent.models.state import GameMode, StructuredGameState
from pokemon_agent.navigation.world_graph import load_world_graph


def _state(**overrides) -> StructuredGameState:
    payload = {
        "map_name": "Red's House 2F",
        "map_id": 0x26,
        "mode": GameMode.OVERWORLD,
        "step": 12,
    }
    payload.update(overrides)
    return StructuredGameState(**payload)


def test_objective_manager_replans_with_flat_objective_payload() -> None:
    captured = {}

    def complete(messages, *, purpose):
        captured["purpose"] = purpose
        payload = json.loads(messages[-1]["content"])
        captured["payload"] = payload
        from pokemon_agent.agent.llm_client import CompletionResponse

        return CompletionResponse(
            content=json.dumps(
                {
                    "goal": "Move toward Oak's Lab",
                    "target_map": "Oak's Lab",
                    "strategy": "Head downstairs, leave the house, and walk north.",
                    "milestone_id": "get_starter",
                    "confidence": 0.9,
                }
            ),
            model="fake",
        )

    manager = ObjectiveManager(
        llm_client=object(),
        context_manager=ContextManager(),
        world_graph=load_world_graph(),
        completion_fn=complete,
    )

    objective = manager.replan(_state(), MemoryState(), stuck_score=0, turn_index=1)
    metadata = manager.last_metadata()

    assert objective.goal == "Move toward Oak's Lab"
    assert objective.target_map == "Oak's Lab"
    assert objective.milestone_id == "get_starter"
    assert captured["purpose"] == "objective planner"
    assert "goal" in captured["payload"]["response_schema"]
    assert metadata.llm_attempted is True
    assert metadata.llm_model == "fake"
    assert metadata.messages


def test_objective_manager_falls_back_when_llm_output_is_invalid() -> None:
    def complete(_messages, *, purpose):
        del purpose
        from pokemon_agent.agent.llm_client import CompletionResponse

        return CompletionResponse(content="not json", model="fake")

    manager = ObjectiveManager(
        llm_client=object(),
        context_manager=ContextManager(),
        world_graph=load_world_graph(),
        completion_fn=complete,
    )

    objective = manager.replan(_state(), MemoryState(), stuck_score=0, turn_index=1)

    assert objective.goal == "Move toward Oak's Lab"
    assert objective.target_map == "Oak's Lab"
    assert objective.strategy


def test_objective_manager_stuck_replan_respects_cooldown() -> None:
    manager = ObjectiveManager(
        llm_client=None,
        context_manager=ContextManager(),
        world_graph=load_world_graph(),
    )
    manager.restore_state(
        StrategicObjective(
            goal="Move toward Oak's Lab",
            target_map="Oak's Lab",
            strategy="Keep moving north.",
            milestone_id="get_starter",
            generated_at_step=5,
        )
    )

    assert manager.should_replan(_state(), stuck_score=8, turn_index=10) is False
    assert manager.should_replan(_state(), stuck_score=8, turn_index=15) is True


def test_objective_manager_replans_when_current_map_is_unknown() -> None:
    manager = ObjectiveManager(
        llm_client=None,
        context_manager=ContextManager(),
        world_graph=load_world_graph(),
    )
    manager.restore_state(
        StrategicObjective(
            goal="Reach Oak's Lab",
            target_map="Oak's Lab",
            strategy="Use the current route.",
            milestone_id="get_starter",
        )
    )

    should_replan = manager.should_replan(
        _state(map_name="Unknown Debug Map", map_id=999),
        stuck_score=0,
        turn_index=3,
    )

    assert should_replan is True
