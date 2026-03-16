from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pokemon_agent.agent.progress import ProgressResult
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.state import StructuredGameState


@dataclass(slots=True)
class StuckState:
    score: int = 0
    steps_since_progress: int = 0
    oscillating: bool = False
    map_oscillating: bool = False


class StuckDetector:
    def __init__(self, threshold: int = 4) -> None:
        self.state = StuckState()
        self.threshold = threshold
        self._last_mode: str | None = None
        self._recent_signatures: deque[tuple] = deque(maxlen=4)
        self._recent_maps: deque[str] = deque(maxlen=4)

    def update(self, game_state: StructuredGameState, action: ActionDecision, progress_classification: str, progress_result: ProgressResult | None = None) -> StuckState:
        del action
        current_mode = game_state.mode.value if game_state.mode else None
        if self._last_mode is not None and current_mode != self._last_mode:
            self.state.score = 0
            self.state.steps_since_progress = 0
            self.state.oscillating = False
            self.state.map_oscillating = False
            self._recent_signatures.clear()
            self._recent_maps.clear()
        self._last_mode = current_mode

        self._recent_signatures.append(game_state.state_signature())
        if game_state.map_name:
            self._recent_maps.append(game_state.map_name)

        repeated_dialogue = progress_result is not None and progress_result.repeated_dialogue
        effective_classification = "no_effect" if repeated_dialogue else progress_classification

        if effective_classification == "no_effect":
            self.state.score += 1
            self.state.steps_since_progress += 1
        elif effective_classification == "regression":
            self.state.score += 2
            self.state.steps_since_progress += 1
        else:
            self.state.score = max(0, self.state.score - (2 if effective_classification == "major_progress" else 1))
            self.state.steps_since_progress = 0

        self.state.oscillating = self._is_oscillating()
        self.state.map_oscillating = self._is_map_oscillating()
        if self.state.oscillating or self.state.map_oscillating or self.state.steps_since_progress >= 6:
            self.state.score = max(self.state.score, self.threshold)
        if len(self._recent_signatures) >= 3 and len(set(self._recent_signatures)) == 1:
            self.state.score = max(self.state.score, self.threshold)
        return self.state

    def restore(self, state: StuckState) -> None:
        self.state = state
        self._recent_signatures.clear()
        self._recent_maps.clear()

    def _is_oscillating(self) -> bool:
        if len(self._recent_signatures) < 4:
            return False
        a, b, c, d = list(self._recent_signatures)
        return a == c and b == d and a != b

    def _is_map_oscillating(self) -> bool:
        maps = list(self._recent_maps)
        if len(maps) < 4:
            return False
        a, b, c, d = maps[-4], maps[-3], maps[-2], maps[-1]
        return a == c and b == d and a != b
