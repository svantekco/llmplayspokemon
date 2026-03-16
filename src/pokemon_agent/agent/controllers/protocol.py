from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.planner import Objective
from pokemon_agent.models.state import StructuredGameState


@dataclass(frozen=True, slots=True)
class NavigationTarget:
    map_name: str | None = None
    x: int | None = None
    y: int | None = None
    landmark_id: str | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class TurnContext:
    objective: Objective | None = None
    navigation_target: NavigationTarget | None = None
    stuck_score: int = 0
    turn_index: int = 0
    previous_action: ActionDecision | None = None
    previous_progress: str | None = None


@runtime_checkable
class Controller(Protocol):
    def step(self, state: StructuredGameState, context: TurnContext) -> PlanningResult:
        """Return the next planning result for the current effective game mode."""
        ...

    def reset(self) -> None:
        """Clear any controller-local state after a mode transition."""
        ...
