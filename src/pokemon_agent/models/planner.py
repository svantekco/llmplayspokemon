from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from pokemon_agent.models.action import ActionDecision, ActionType


class ObjectiveHorizon(str, Enum):
    SHORT_TERM = "short"
    MID_TERM = "mid"
    LONG_TERM = "long"


class ObjectiveStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    INVALIDATED = "invalidated"


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
    summary: str
    priority: int = 0
    target: ObjectiveTarget | None = None
    success_conditions: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    status: ObjectiveStatus = ObjectiveStatus.ACTIVE
    source: str = "rule"


class CandidateNextStep(BaseModel):
    id: str
    type: str
    target: ObjectiveTarget | None = None
    why: str
    priority: int = 0
    expected_success_signal: str
    objective_id: str | None = None
    action: ActionDecision | None = Field(default=None, exclude=True)
    target_x: int | None = Field(default=None, exclude=True)
    target_y: int | None = Field(default=None, exclude=True)
    follow_up_action: ActionType | None = Field(default=None, exclude=True)
    step_budget: int = Field(default=1, ge=1, exclude=True)


class PlannerDecision(BaseModel):
    candidate_id: str | None = None
    intent: str | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _validate_choice(self) -> "PlannerDecision":
        if not self.candidate_id and not self.intent:
            raise ValueError("PlannerDecision requires candidate_id or intent")
        return self


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

