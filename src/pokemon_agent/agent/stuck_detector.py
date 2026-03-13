from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.state import StructuredGameState


@dataclass(slots=True)
class StuckState:
    score: int = 0
    recent_signatures: deque = field(default_factory=lambda: deque(maxlen=8))
    recent_failed_actions: list[str] = field(default_factory=list)
    loop_signature: str | None = None
    recovery_hint: str | None = None
    repeated_state_count: int = 0
    steps_since_progress: int = 0
    oscillating: bool = False


class StuckDetector:
    def __init__(self, threshold: int = 4) -> None:
        self.state = StuckState()
        self.threshold = threshold

    def update(self, game_state: StructuredGameState, action: ActionDecision, progress_classification: str) -> StuckState:
        signature = game_state.state_signature()
        self.state.recent_signatures.append(signature)

        progress_made = progress_classification in {
            "major_progress",
            "movement_success",
            "interaction_success",
            "partial_progress",
        }

        if progress_classification == "no_effect":
            self.state.score += 1
            self.state.recent_failed_actions.append(action.action.value)
            self.state.recent_failed_actions = self.state.recent_failed_actions[-5:]
            self.state.steps_since_progress += 1
        elif progress_classification == "regression":
            self.state.score += 2
            self.state.steps_since_progress += 1
        else:
            self.state.score = max(0, self.state.score - (2 if progress_classification == "major_progress" else 1))
            if progress_made:
                self.state.steps_since_progress = 0

        self.state.repeated_state_count = sum(1 for item in self.state.recent_signatures if item == signature)
        self.state.oscillating = self._is_oscillating()
        if self.state.repeated_state_count >= 3 or self.state.oscillating or self.state.steps_since_progress >= 6:
            self.state.score = max(self.state.score, self.threshold)

        if len(self.state.recent_signatures) >= 4:
            last_four = list(self.state.recent_signatures)[-4:]
            self.state.loop_signature = " -> ".join(
                f"{item[1]}({item[2]},{item[3]})/{item[5]}" for item in last_four
            )
        else:
            self.state.loop_signature = None

        self.state.recovery_hint = self._build_recovery_hint(game_state, action, progress_classification)
        return self.state

    def restore(self, state: StuckState) -> None:
        self.state = state

    def _is_oscillating(self) -> bool:
        if len(self.state.recent_signatures) < 4:
            return False
        a, b, c, d = list(self.state.recent_signatures)[-4:]
        return a == c and b == d and a != b

    def _build_recovery_hint(
        self,
        game_state: StructuredGameState,
        action: ActionDecision,
        progress_classification: str,
    ) -> str | None:
        if game_state.text_box_open:
            return "Text is open; prefer PRESS_A to advance dialogue."
        if game_state.menu_open:
            return "A menu is open; use PRESS_B to back out or a single direction to navigate."
        if game_state.battle_state:
            return "Battle is active; keep inputs single-step and prefer PRESS_A unless the menu changes."
        if progress_classification == "no_effect" and action.action.value.startswith("MOVE_"):
            return "Movement had no effect; try a different direction or interact with PRESS_A."
        if self.state.oscillating:
            return "You are oscillating between states; stop repeating the same pattern."
        if self.state.score >= self.threshold:
            return "Stuck score is high; prefer a recovery action over repeating the last failed input."
        return None
