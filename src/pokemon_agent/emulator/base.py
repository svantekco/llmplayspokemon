from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Any
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.state import StructuredGameState


class EmulatorAdapter(ABC):
    @abstractmethod
    def get_raw_state(self):
        raise NotImplementedError

    @abstractmethod
    def get_structured_state(self) -> StructuredGameState:
        raise NotImplementedError

    @abstractmethod
    def press_button(self, button: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def advance_frames(self, n: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute_action(self, action: ActionDecision) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def save_state(self, path: str | Path) -> None:
        raise NotImplementedError("save_state is not implemented for this adapter")

    def load_state(self, path: str | Path) -> None:
        raise NotImplementedError("load_state is not implemented for this adapter")

    def begin_planning_wait(self) -> None:
        return None

    def pump_planning_wait(self) -> None:
        time.sleep(0.01)

    def end_planning_wait(self) -> None:
        return None

    def capture_screen_image(self) -> Any | None:
        return None

    def set_live_path_overlay(
        self,
        state: StructuredGameState,
        suggested_path: list[tuple[int, int]],
    ) -> None:
        del state, suggested_path
        return None

    def clear_live_path_overlay(self) -> None:
        return None
