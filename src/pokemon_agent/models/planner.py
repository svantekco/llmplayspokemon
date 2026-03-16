from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from pokemon_agent.models.action import ActionDecision, Task


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


class StrategicObjective(BaseModel):
    goal: str
    target_map: str | None = None
    target_landmark: str | None = None
    target_landmark_type: str | None = None
    strategy: str
    milestone_id: str | None = None
    confidence: float = 0.8
    generated_at_step: int = 0
    generated_at_map: str | None = None

    @classmethod
    def from_legacy_payload(cls, payload: dict[str, Any]) -> "StrategicObjective | None":
        if "human_plan" not in payload or "internal_plan" not in payload:
            return None
        human_plan = payload.get("human_plan") or {}
        internal_plan = payload.get("internal_plan") or {}
        plan_type = str(internal_plan.get("plan_type") or "").strip().lower()
        if plan_type in {"recover", "advance_dialogue", "resolve_menu", "battle_default"}:
            return None
        goal = (
            str(human_plan.get("short_term_goal") or "").strip()
            or str(human_plan.get("mid_term_goal") or "").strip()
            or str(human_plan.get("long_term_goal") or "").strip()
        )
        strategy = (
            str(human_plan.get("current_strategy") or "").strip()
            or str(internal_plan.get("notes") or "").strip()
            or goal
        )
        if not goal or not strategy:
            return None
        confidence = internal_plan.get("confidence")
        try:
            normalized_confidence = float(confidence) if confidence is not None else 0.8
        except (TypeError, ValueError):
            normalized_confidence = 0.8
        generated_at_step = payload.get("generated_at_step")
        try:
            normalized_step = int(generated_at_step) if generated_at_step is not None else 0
        except (TypeError, ValueError):
            normalized_step = 0
        return cls(
            goal=goal,
            target_map=internal_plan.get("target_map_name"),
            target_landmark=internal_plan.get("target_landmark_id"),
            target_landmark_type=internal_plan.get("target_landmark_type"),
            strategy=strategy,
            milestone_id=payload.get("valid_for_milestone_id"),
            confidence=normalized_confidence,
            generated_at_step=normalized_step,
            generated_at_map=payload.get("valid_for_map_name"),
        )


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

    def to_strategic_objective(self) -> StrategicObjective | None:
        return StrategicObjective.from_legacy_payload(self.model_dump(mode="json", exclude_none=True))

    @classmethod
    def from_strategic_objective(cls, objective: StrategicObjective) -> "ObjectivePlanEnvelope":
        plan_type = "go_to_landmark" if objective.target_landmark else "go_to_map"
        return cls(
            human_plan=HumanObjectivePlan(
                short_term_goal=objective.goal,
                mid_term_goal=objective.goal,
                long_term_goal=objective.goal,
                current_strategy=objective.strategy,
            ),
            internal_plan=InternalObjectivePlan(
                plan_type=plan_type,
                target_map_name=objective.target_map,
                target_landmark_id=objective.target_landmark,
                target_landmark_type=objective.target_landmark_type,
                confidence=objective.confidence,
                notes=objective.strategy,
            ),
            generated_at_step=objective.generated_at_step,
            valid_for_map_name=objective.generated_at_map,
            valid_for_milestone_id=objective.milestone_id,
        )


@dataclass(slots=True)
class CandidateRuntime:
    action: ActionDecision | None = None
    task: Task | None = None
    follow_up_task: Task | None = None


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
