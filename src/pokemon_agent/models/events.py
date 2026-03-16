from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class EventType(str, Enum):
    MOVED = "MOVED"
    BLOCKED_MOVE = "BLOCKED_MOVE"
    MAP_CHANGED = "MAP_CHANGED"
    MENU_OPENED = "MENU_OPENED"
    MENU_CLOSED = "MENU_CLOSED"
    TEXT_OPENED = "TEXT_OPENED"
    TEXT_ADVANCED = "TEXT_ADVANCED"
    BATTLE_STARTED = "BATTLE_STARTED"
    BATTLE_ENDED = "BATTLE_ENDED"
    ITEM_RECEIVED = "ITEM_RECEIVED"
    CONNECTOR_CONFIRMED = "CONNECTOR_CONFIRMED"
    MILESTONE_COMPLETE = "MILESTONE_COMPLETE"
    GOAL_UPDATED = "GOAL_UPDATED"
    NO_EFFECT = "NO_EFFECT"
    UNKNOWN = "UNKNOWN"


class EventRecord(BaseModel):
    type: EventType
    summary: str
    step: int
    metadata: dict = Field(default_factory=dict)
