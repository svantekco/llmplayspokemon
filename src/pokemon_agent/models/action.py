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
