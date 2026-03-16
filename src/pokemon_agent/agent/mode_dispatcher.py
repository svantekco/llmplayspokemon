from __future__ import annotations

from pokemon_agent.agent.controllers.protocol import Controller
from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.controllers.recovery import RecoveryController
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState


class ModeDispatcher:
    def __init__(
        self,
        controllers: dict[GameMode, Controller],
        *,
        recovery_controller: RecoveryController | None = None,
    ) -> None:
        missing_modes = [mode.value for mode in GameMode if mode not in controllers]
        if missing_modes:
            missing_text = ", ".join(sorted(missing_modes))
            raise ValueError(f"Missing controller registrations for: {missing_text}")
        self._controllers = dict(controllers)
        self._recovery_controller = recovery_controller
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
            if self._recovery_controller is not None:
                self._recovery_controller.reset()
        if self._recovery_controller is not None:
            recovery = self._recovery_controller.step(state, context)
            if recovery is not None:
                self._last_mode = mode
                return recovery
        controller = self._controllers.get(mode)
        if controller is None:
            raise ValueError(f"No controller registered for mode {mode.value}")
        self._last_mode = mode
        return controller.step(state, context)
