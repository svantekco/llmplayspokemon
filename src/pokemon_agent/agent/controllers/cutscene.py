from __future__ import annotations

from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import StructuredGameState


class CutsceneController:
    def step(self, state: StructuredGameState, context: TurnContext) -> PlanningResult:
        del state, context
        return PlanningResult(
            action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="advance cutscene"),
            planner_source="cutscene_controller",
        )

    def reset(self) -> None:
        return None
