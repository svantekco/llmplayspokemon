from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pokemon_agent.agent.llm_client import LLMUsage
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import Task
from pokemon_agent.models.planner import CandidateNextStep


@dataclass(slots=True)
class PlanningResult:
    action: ActionDecision | None = None
    task: Task | None = None
    follow_up_task: Task | None = None
    raw_response: str | None = None
    used_fallback: bool = False
    planner_source: str = "fallback"
    messages: list[dict[str, Any]] = field(default_factory=list)
    prompt_metrics: dict[str, Any] | None = None
    llm_usage: LLMUsage | None = None
    llm_attempted: bool = False
    llm_model: str | None = None
    objective_id: str | None = None
    candidate_id: str | None = None
    candidates: list[CandidateNextStep] = field(default_factory=list)
    executor_task: Task | None = None
    suggested_path: list[tuple[int, int]] = field(default_factory=list)
