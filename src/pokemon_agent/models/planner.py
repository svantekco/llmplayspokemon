from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from pokemon_agent.models.action import ActionDecision, ActionType


class ObjectiveHorizon(str, Enum):
    SHORT_TERM = "short"
    MID_TERM = "mid"
    LONG_TERM = "long"


class ObjectiveTarget(BaseModel):
    kind: str
    map_id: str | int | None = None
    map_name: str | None = None
    x: int | None = None
    y: int | None = None
    detail: str | None = None


class Objective(BaseModel):
    id: str
    horizon: ObjectiveHorizon
    target: ObjectiveTarget | None = None


@dataclass(slots=True)
class CandidateRuntime:
    action: ActionDecision | None = None
    target_x: int | None = None
    target_y: int | None = None
    follow_up_action: ActionType | None = None
    step_budget: int = 1


class CandidateNextStep(BaseModel):
    id: str
    type: str
    target: ObjectiveTarget | None = None
    why: str
    priority: int = 0
    expected_success_signal: str
    objective_id: str | None = None


class PlannerDecision(BaseModel):
    candidate_id: str
    reason: str = ""


class ExecutionPlan(BaseModel):
    objective_id: str | None = None
    candidate_id: str
    plan_type: str
    target: ObjectiveTarget | None = None
    expected_success_signal: str
    step_budget: int = Field(default=1, ge=1)
    button_action: ActionType | None = None
    target_x: int | None = None
    target_y: int | None = None
    follow_up_action: ActionType | None = None
    map_id: str | int | None = None
    collision_hash: str | None = None
    reason: str = ""
    started_step: int | None = None
