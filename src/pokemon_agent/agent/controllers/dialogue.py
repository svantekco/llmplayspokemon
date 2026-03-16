from __future__ import annotations

from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import StructuredGameState

_YES_KEYWORDS = (
    "heal",
    "restore your pokemon",
    "take",
    "accept",
    "want",
    "receive",
    "board",
    "teach",
    "use",
    "buy",
    "give",
)
_NO_KEYWORDS = ("save", "quit", "cancel", "stop", "give up")
_SHOP_EXIT_KEYWORDS = ("is there anything", "anything else")


class DialogueController:
    def step(self, state: StructuredGameState, context: TurnContext) -> PlanningResult:
        del context
        if not state.text_box_open:
            return PlanningResult(
                action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="clear text transition"),
                planner_source="dialogue_controller",
            )

        choice = self._deterministic_choice(state)
        if choice is not None:
            action = self._yes_no_action(choice=choice, current_cursor=state.metadata.get("cursor"))
            return PlanningResult(action=action, planner_source="dialogue_controller")

        if bool(state.metadata.get("yes_no_prompt")):
            action = self._yes_no_action(
                choice="YES",
                current_cursor=state.metadata.get("cursor"),
                reason="default to yes for unknown prompt",
            )
            return PlanningResult(action=action, planner_source="dialogue_controller")

        return PlanningResult(
            action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="advance dialogue"),
            planner_source="dialogue_controller",
        )

    def reset(self) -> None:
        return None

    def _deterministic_choice(self, state: StructuredGameState) -> str | None:
        if not bool(state.metadata.get("yes_no_prompt")):
            return None
        dialogue = str(state.metadata.get("dialogue_text") or state.metadata.get("dialogue") or "").lower()
        if not dialogue:
            return None
        if any(keyword in dialogue for keyword in _SHOP_EXIT_KEYWORDS):
            return "NO"
        choose_yes = any(keyword in dialogue for keyword in _YES_KEYWORDS)
        choose_no = any(keyword in dialogue for keyword in _NO_KEYWORDS)
        if choose_yes == choose_no:
            return None
        return "YES" if choose_yes else "NO"

    def _yes_no_action(
        self,
        *,
        choice: str,
        current_cursor: object,
        reason: str | None = None,
    ) -> ActionDecision:
        cursor = str(current_cursor or "YES").upper()
        target = choice.upper()
        if cursor == target:
            return ActionDecision(
                action=ActionType.PRESS_A,
                repeat=1,
                reason=reason or f"confirm {target.lower()}",
            )
        if target == "YES":
            return ActionDecision(
                action=ActionType.MOVE_UP,
                repeat=1,
                reason=reason or "move to yes",
            )
        return ActionDecision(
            action=ActionType.MOVE_DOWN,
            repeat=1,
            reason=reason or "move to no",
        )
