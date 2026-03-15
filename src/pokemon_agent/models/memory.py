from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .action import ActionType
from .events import EventRecord
from .planner import Objective
from .planner import ObjectivePlanEnvelope
from .state import WorldCoordinate


class GoalStack(BaseModel):
    long_term_goal: str = "Complete Pokemon Red main story"
    mid_term_goal: str = "Leave the starting area and begin reliable overworld progress"
    short_term_goal: str = "Establish a stable local loop and explore the current map"
    current_strategy: str = "Prefer small validated actions and react to meaningful state changes"
    success_conditions: list[str] = Field(default_factory=list)
    active_objectives: list[Objective] = Field(default_factory=list)


class ConnectorStatus(str, Enum):
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"


class DiscoveredConnector(BaseModel):
    id: str
    source_map: str
    source_side: str | None = None
    source_x: int | None = None
    source_y: int | None = None
    kind: str = "unknown"
    activation_mode: str | None = None
    status: ConnectorStatus = ConnectorStatus.SUSPECTED
    approach_x: int | None = None
    approach_y: int | None = None
    transition_action: ActionType | None = None
    destination_map: str | None = None
    destination_x: int | None = None
    destination_y: int | None = None
    discovered_step: int | None = None
    confirmed_step: int | None = None


class DiscoveredMap(BaseModel):
    map_name: str
    map_id: str | int | None = None
    walkable: list[WorldCoordinate] = Field(default_factory=list)
    blocked: list[WorldCoordinate] = Field(default_factory=list)
    connectors: list[str] = Field(default_factory=list)
    last_seen_step: int | None = None


class NavigationGoal(BaseModel):
    target_map_name: str
    final_target_map_name: str | None = None
    target_landmark_id: str | None = None
    target_landmark_type: str | None = None
    source: str = "objective"
    objective_kind: str = "reach_boundary_side"
    engine_mode: str = "progression"
    current_map_name: str | None = None
    next_map_name: str | None = None
    next_hop_kind: str | None = None
    next_hop_side: str | None = None
    target_connector_id: str | None = None
    failed_candidate_ids: list[str] = Field(default_factory=list)
    failed_connector_ids: list[str] = Field(default_factory=list)
    failed_sides: list[str] = Field(default_factory=list)
    last_candidate_id: str | None = None
    started_step: int | None = None
    last_confirmed_step: int | None = None
    confirmation_required_map: str | None = None


class WorldMapMemory(BaseModel):
    maps: dict[str, DiscoveredMap] = Field(default_factory=dict)
    connectors: dict[str, DiscoveredConnector] = Field(default_factory=dict)


class LongTermKnowledge(BaseModel):
    known_locations: dict[str, str] = Field(default_factory=dict)
    story_flags: list[str] = Field(default_factory=list)
    navigation_notes: list[str] = Field(default_factory=list)
    heuristics: list[str] = Field(default_factory=list)
    world_map: WorldMapMemory = Field(default_factory=WorldMapMemory)
    navigation_goal: NavigationGoal | None = None
    objective_plan: ObjectivePlanEnvelope | None = None


class MemoryState(BaseModel):
    recent_events: list[EventRecord] = Field(default_factory=list)
    goals: GoalStack = Field(default_factory=GoalStack)
    long_term: LongTermKnowledge = Field(default_factory=LongTermKnowledge)
