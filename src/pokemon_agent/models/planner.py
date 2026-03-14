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


class ObjectivePlanStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    INVALID = "invalid"


class HumanObjectivePlan(BaseModel):
    short_term_goal: str
    mid_term_goal: str
    long_term_goal: str
    current_strategy: str


class InternalObjectivePlan(BaseModel):
    plan_type: str
    target_map_name: str | None = None
    target_landmark_id: str | None = None
    target_landmark_type: str | None = None
    target_npc_hint: str | None = None
    success_signal: str | None = None
    stop_when: str | None = None
    confidence: float | None = None
    notes: str | None = None


class ObjectivePlanEnvelope(BaseModel):
    human_plan: HumanObjectivePlan
    internal_plan: InternalObjectivePlan
    status: ObjectivePlanStatus = ObjectivePlanStatus.ACTIVE
    generated_at_step: int | None = None
    valid_for_milestone_id: str | None = None
    valid_for_map_name: str | None = None
    replan_reason: str | None = None


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
    distance: int | None = None
    advances_target: bool = False
    backtrack: bool = False
    blacklisted: bool = False


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
