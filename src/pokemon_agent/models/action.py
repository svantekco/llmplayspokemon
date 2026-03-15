from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_TO_COORDINATE = "MOVE_TO_COORDINATE"
    PRESS_A = "PRESS_A"
    PRESS_B = "PRESS_B"
    PRESS_START = "PRESS_START"


class ActionDecision(BaseModel):
    action: ActionType
    repeat: int = Field(default=1, ge=1, le=8)
    reason: str = ""
    target_x: int | None = None
    target_y: int | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_coordinate_targets(self) -> "ActionDecision":
        if self.action == ActionType.MOVE_TO_COORDINATE:
            if self.target_x is None or self.target_y is None:
                raise ValueError("MOVE_TO_COORDINATE requires target_x and target_y")
        elif self.target_x is not None or self.target_y is not None:
            raise ValueError("target_x and target_y are only valid for MOVE_TO_COORDINATE")
        return self


class TaskKind(str, Enum):
    NAVIGATE_TO = "navigate_to"
    NAVIGATE_ADJACENT = "navigate_adjacent"
    INTERACT = "interact"
    PRESS_BUTTON = "press_button"
    ENTER_CONNECTOR = "enter_connector"
    WALK_BOUNDARY = "walk_boundary"


class Task(BaseModel):
    kind: TaskKind
    reason: str = ""
    target_x: int | None = None
    target_y: int | None = None
    connector_id: str | None = None
    direction: str | None = None
    button: ActionType | None = None
    repeat: int = Field(default=1, ge=1, le=8)

    @model_validator(mode="after")
    def _validate_task_fields(self) -> "Task":
        coordinate_required = {
            TaskKind.NAVIGATE_TO,
            TaskKind.NAVIGATE_ADJACENT,
        }
        if self.kind in coordinate_required and (self.target_x is None or self.target_y is None):
            raise ValueError(f"{self.kind.value} requires target_x and target_y")
        if self.kind == TaskKind.ENTER_CONNECTOR and self.connector_id is None:
            raise ValueError("enter_connector requires connector_id")
        if self.kind == TaskKind.WALK_BOUNDARY and self.direction is None:
            raise ValueError("walk_boundary requires direction")
        if self.kind == TaskKind.PRESS_BUTTON and self.button is None:
            raise ValueError("press_button requires button")
        return self


class ExecutorStatus(str, Enum):
    STEPPING = "STEPPING"
    DONE = "DONE"
    BLOCKED = "BLOCKED"
    INTERRUPTED = "INTERRUPTED"


class StepResult(BaseModel):
    status: ExecutorStatus
    action: ActionDecision | None = None
    blocked_reason: str | None = None
    suggested_path: list[tuple[int, int]] = Field(default_factory=list)
