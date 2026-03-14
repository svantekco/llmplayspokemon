from __future__ import annotations

from dataclasses import dataclass, field
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import StructuredGameState


@dataclass(slots=True)
class ProgressResult:
    classification: str
    changed_fields: list[str] = field(default_factory=list)
    newly_completed_subgoals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # True when the same dialogue/sign was opened again — should not count as real progress.
    repeated_dialogue: bool = False


class ProgressDetector:
    def __init__(self) -> None:
        # Dialogue text from the most recently closed text box, used to detect sign re-reads.
        self._last_closed_dialogue: str | None = None

    def compare(self, previous: StructuredGameState, current: StructuredGameState) -> ProgressResult:
        changed: list[str] = []
        notes: list[str] = []
        completed: list[str] = []

        previous_bootstrap = previous.is_bootstrap()
        current_bootstrap = current.is_bootstrap()
        if previous_bootstrap or current_bootstrap:
            if previous.bootstrap_phase() != current.bootstrap_phase():
                changed.append("bootstrap_phase")
                label = current.bootstrap_phase() or current.map_name
                notes.append(f"Bootstrap phase changed to {label}")
            if previous_bootstrap != current_bootstrap:
                changed.append("engine_phase")
                notes.append("Reached active gameplay" if not current_bootstrap else "Entered bootstrap phase")
            if not changed:
                return ProgressResult("no_effect", [], [], ["No meaningful state change detected"])
            if previous_bootstrap and not current_bootstrap:
                completed.append(f"Reached {current.map_name}")
                return ProgressResult("major_progress", changed, completed, notes)
            return ProgressResult("interaction_success", changed, completed, notes)

        if previous.map_id != current.map_id or previous.map_name != current.map_name:
            changed.extend(["map_id", "map_name"])
            notes.append(f"Map changed to {current.map_name}")
            completed.append(f"Entered {current.map_name}")
        if previous.x != current.x or previous.y != current.y:
            changed.append("position")
            notes.append("Position changed")
        if previous.facing != current.facing:
            changed.append("facing")
        if previous.menu_open != current.menu_open:
            changed.append("menu_open")
            notes.append("Menu visibility changed")
        if previous.text_box_open != current.text_box_open:
            changed.append("text_box_open")
            notes.append("Text box visibility changed")
        if previous.mode != current.mode:
            changed.append("mode")
            notes.append(f"Mode changed to {current.mode.value}")
        if previous.battle_state != current.battle_state:
            changed.append("battle_state")
            notes.append("Battle state changed")
        if self._screen_changed(previous, current):
            changed.append("screen_state")
            notes.append("Screen content changed")

        inventory_gain = self._inventory_gain(previous, current)
        if inventory_gain:
            changed.append("inventory")
            notes.append(f"Inventory changed: {', '.join(inventory_gain)}")
            completed.extend([f"Obtained {item}" for item in inventory_gain])

        # Detect when the same dialogue reopens (sign re-read loop).
        repeated_dialogue = False
        if not previous.text_box_open and current.text_box_open:
            current_dialogue = self._get_dialogue(current)
            if current_dialogue and current_dialogue == self._last_closed_dialogue:
                repeated_dialogue = True
        if previous.text_box_open and not current.text_box_open:
            self._last_closed_dialogue = self._get_dialogue(previous)

        if not changed:
            return ProgressResult("no_effect", [], [], ["No meaningful state change detected"])
        if previous.battle_state and not current.battle_state:
            return ProgressResult("major_progress", changed, completed, notes or ["Battle ended"])
        if "map_id" in changed or "inventory" in changed:
            return ProgressResult("major_progress", changed, completed, notes)
        if "position" in changed:
            return ProgressResult("movement_success", changed, completed, notes)
        if "text_box_open" in changed or "menu_open" in changed or "battle_state" in changed or "screen_state" in changed:
            return ProgressResult("interaction_success", changed, completed, notes, repeated_dialogue=repeated_dialogue)
        if current.mode == GameMode.BATTLE and previous.mode != GameMode.BATTLE:
            return ProgressResult("interaction_success", changed, completed, notes)
        if "facing" in changed or "mode" in changed:
            return ProgressResult("partial_progress", changed, completed, notes)
        return ProgressResult("unknown", changed, completed, notes or ["State changed in an unclassified way"])

    @staticmethod
    def _get_dialogue(state: StructuredGameState) -> str | None:
        text = state.metadata.get("dialogue_text") or state.metadata.get("dialogue")
        if isinstance(text, str):
            return text.strip() or None
        return None

    def _inventory_gain(self, previous: StructuredGameState, current: StructuredGameState) -> list[str]:
        before = {item.name: item.count for item in previous.inventory}
        after = {item.name: item.count for item in current.inventory}
        gained: list[str] = []
        for name, count in after.items():
            if count > before.get(name, 0):
                gained.append(name)
        return gained

    def _screen_changed(self, previous: StructuredGameState, current: StructuredGameState) -> bool:
        before_tile_hash = previous.metadata.get("tile_hash")
        after_tile_hash = current.metadata.get("tile_hash")
        if before_tile_hash and after_tile_hash and before_tile_hash != after_tile_hash:
            if previous.text_box_open and current.text_box_open:
                return True
            if previous.menu_open and current.menu_open:
                return True
            if previous.mode == GameMode.BATTLE and current.mode == GameMode.BATTLE:
                return True
        before_dialogue = previous.metadata.get("dialogue_text", previous.metadata.get("dialogue"))
        after_dialogue = current.metadata.get("dialogue_text", current.metadata.get("dialogue"))
        return bool(
            previous.text_box_open
            and current.text_box_open
            and before_dialogue is not None
            and before_dialogue != after_dialogue
        )
