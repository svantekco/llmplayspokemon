from __future__ import annotations

from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.menu_manager import MenuManager
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import StructuredGameState


class MenuController:
    def __init__(self, menu_manager: MenuManager | None = None) -> None:
        self._menu_manager = menu_manager or MenuManager()

    def step(self, state: StructuredGameState, context: TurnContext) -> PlanningResult:
        if not state.menu_open:
            return PlanningResult(
                action=ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="clear menu transition"),
                planner_source="menu_controller",
            )

        objective_id = context.objective.id if context.objective is not None else "menu_controller"
        candidates = self._menu_manager.build_candidates(state, objective_id)
        runtime = self._menu_manager.runtime_map()
        if candidates:
            best = sorted(candidates, key=lambda item: (-item.priority, item.id))[0]
            selected = runtime.get(best.id)
            if selected is not None and selected.action is not None:
                action = selected.action.model_copy(deep=True)
                if not action.reason:
                    action.reason = best.why
                return PlanningResult(action=action, planner_source="menu_controller")

        return PlanningResult(
            action=ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="close menu"),
            planner_source="menu_controller",
        )

    def reset(self) -> None:
        self._menu_manager.reset()
