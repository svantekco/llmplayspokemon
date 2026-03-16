from __future__ import annotations

from collections.abc import Callable

from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.models.state import StructuredGameState

ControllerStep = Callable[[StructuredGameState, TurnContext], PlanningResult]


class _BaseStubController:
    def __init__(self, step_fn: ControllerStep) -> None:
        self._step_fn = step_fn

    def step(self, state: StructuredGameState, context: TurnContext) -> PlanningResult:
        return self._step_fn(state, context)

    def reset(self) -> None:
        return None


class StubOverworldController(_BaseStubController):
    pass


class StubMenuController(_BaseStubController):
    pass


class StubTextController(_BaseStubController):
    pass


class StubBattleController(_BaseStubController):
    pass


class StubCutsceneController(_BaseStubController):
    pass


class StubUnknownController(_BaseStubController):
    pass
