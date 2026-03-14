from __future__ import annotations

import json
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.agent.stuck_detector import StuckState
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState


class ActionValidator:
    def __init__(self, max_repeat: int = 4) -> None:
        self.max_repeat = max_repeat

    def parse(self, raw_text: str) -> ActionDecision:
        payload = self._extract_json(raw_text)
        data = json.loads(payload)
        if isinstance(data, dict) and "repeat" in data:
            data["repeat"] = min(int(data["repeat"]), self.max_repeat, 8)
        decision = ActionDecision.model_validate(data)
        return self.validate(decision)

    def validate(self, decision: ActionDecision, state: StructuredGameState | None = None) -> ActionDecision:
        decision = decision.model_copy(deep=True)
        if decision.repeat > self.max_repeat:
            decision.repeat = self.max_repeat
        if decision.action == ActionType.MOVE_TO_COORDINATE:
            decision.repeat = 1
        if state and (state.menu_open or state.text_box_open or state.mode in {GameMode.BATTLE, GameMode.TEXT, GameMode.MENU}):
            if decision.action == ActionType.MOVE_TO_COORDINATE:
                return self.fallback(state, reason=decision.reason or "coordinate routing is only valid in overworld")
            decision.repeat = 1
        if state and state.is_bootstrap() and decision.action == ActionType.MOVE_TO_COORDINATE:
            return self.bootstrap(state, reason=decision.reason or "coordinate routing is not valid during bootstrap")
        return decision

    def fallback(
        self,
        state: StructuredGameState,
        stuck_state: StuckState | None = None,
        reason: str = "deterministic fallback",
    ) -> ActionDecision:
        if state.is_bootstrap():
            return self.bootstrap(state, stuck_state=stuck_state, reason=reason)

        stuck_score = stuck_state.score if stuck_state else 0
        recent_failed = set(stuck_state.recent_failed_actions[-3:] if stuck_state else [])

        if state.text_box_open:
            return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason=reason)
        if state.menu_open:
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason=reason)
        if state.mode == GameMode.BATTLE:
            return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason=reason)
        if stuck_score >= 4 and "PRESS_A" not in recent_failed:
            return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason=reason)

        cycle = [
            ActionType.MOVE_UP,
            ActionType.MOVE_RIGHT,
            ActionType.MOVE_DOWN,
            ActionType.MOVE_LEFT,
        ]
        candidates = [action for action in cycle if action.value not in recent_failed]
        if candidates:
            action = candidates[stuck_score % len(candidates)]
            return ActionDecision(action=action, repeat=1, reason=reason)

        if stuck_score >= 6 and "PRESS_B" not in recent_failed:
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason=reason)
        if stuck_score >= 8 and "PRESS_START" not in recent_failed:
            return ActionDecision(action=ActionType.PRESS_START, repeat=1, reason=reason)

        action = cycle[stuck_score % len(cycle)]
        return ActionDecision(action=action, repeat=1, reason=reason)

    def bootstrap(
        self,
        state: StructuredGameState,
        stuck_state: StuckState | None = None,
        reason: str = "deterministic startup bootstrap",
    ) -> ActionDecision:
        recent_failed = set(stuck_state.recent_failed_actions[-3:] if stuck_state else [])
        phase = state.bootstrap_phase()

        if phase == "title_menu" or phase == "intro_cutscene" or state.text_box_open:
            return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason=reason)

        if phase == "title_screen":
            primary = ActionType.PRESS_START
            secondary = ActionType.PRESS_A
        else:
            primary = ActionType.PRESS_START
            secondary = ActionType.PRESS_A

        action = primary if primary.value not in recent_failed else secondary
        return ActionDecision(action=action, repeat=1, reason=reason)

    def _extract_json(self, raw_text: str) -> str:
        text = raw_text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Model response did not contain a JSON object")
        return text[start : end + 1]
