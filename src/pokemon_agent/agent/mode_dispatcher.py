from __future__ import annotations

from pokemon_agent.agent.controllers.protocol import Controller
from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState


class ModeDispatcher:
    def __init__(self, controllers: dict[GameMode, Controller]) -> None:
        missing_modes = [mode.value for mode in GameMode if mode not in controllers]
        if missing_modes:
            missing_text = ", ".join(sorted(missing_modes))
            raise ValueError(f"Missing controller registrations for: {missing_text}")
        self._controllers = dict(controllers)
        self._last_mode: GameMode | None = None

    @staticmethod
    def effective_mode(state: StructuredGameState) -> GameMode:
        if state.battle_state is not None:
            return GameMode.BATTLE
        if state.menu_open:
            return GameMode.MENU
        if state.text_box_open:
            return GameMode.TEXT
        return state.mode

    def dispatch(self, state: StructuredGameState, context: TurnContext) -> PlanningResult:
        mode = self.effective_mode(state)
        if self._last_mode is not None and mode != self._last_mode:
            previous = self._controllers.get(self._last_mode)
            if previous is not None:
                previous.reset()
        controller = self._controllers.get(mode)
        if controller is None:
            raise ValueError(f"No controller registered for mode {mode.value}")
        self._last_mode = mode
        return controller.step(state, context)
