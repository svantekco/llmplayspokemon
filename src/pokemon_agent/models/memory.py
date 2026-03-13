from __future__ import annotations

from pydantic import BaseModel, Field
from .events import EventRecord
from .planner import Objective


class GoalStack(BaseModel):
    long_term_goal: str = "Complete Pokemon Red main story"
    mid_term_goal: str = "Leave the starting area and begin reliable overworld progress"
    short_term_goal: str = "Establish a stable local loop and explore the current map"
    current_strategy: str = "Prefer small validated actions and react to meaningful state changes"
    success_conditions: list[str] = Field(default_factory=list)
    active_objectives: list[Objective] = Field(default_factory=list)


class LongTermKnowledge(BaseModel):
    known_locations: dict[str, str] = Field(default_factory=dict)
    story_flags: list[str] = Field(default_factory=list)
    navigation_notes: list[str] = Field(default_factory=list)
    heuristics: list[str] = Field(default_factory=list)


class MemoryState(BaseModel):
    recent_events: list[EventRecord] = Field(default_factory=list)
    goals: GoalStack = Field(default_factory=GoalStack)
    long_term: LongTermKnowledge = Field(default_factory=LongTermKnowledge)
