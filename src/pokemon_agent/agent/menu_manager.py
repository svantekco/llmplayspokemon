from __future__ import annotations

from dataclasses import dataclass
import re

from pokemon_agent.data.walkthrough import Milestone
from pokemon_agent.data.walkthrough import get_current_milestone
from pokemon_agent.emulator.text_reader import POKEMON_RED_CHAR_MAP
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.planner import CandidateNextStep
from pokemon_agent.models.planner import CandidateRuntime
from pokemon_agent.models.planner import ObjectiveTarget
from pokemon_agent.models.state import StructuredGameState

START_MENU_TYPE = "START_MENU"
ITEM_MENU_TYPE = "ITEM_MENU"
PARTY_MENU_TYPE = "PARTY_MENU"
POKEMON_SUBMENU_TYPE = "POKEMON_SUBMENU"
SHOP_MENU_TYPE = "SHOP_MENU"
SHOP_BUY_MENU_TYPE = "SHOP_BUY_MENU"
GENERIC_MENU_TYPE = "GENERIC_MENU"
UNKNOWN_MENU_TYPE = "UNKNOWN_MENU"

START_MENU_KEYWORDS = {"POKEDEX", "POKEMON", "ITEM", "SAVE", "OPTION", "EXIT"}
SHOP_MENU_KEYWORDS = {"BUY", "SELL", "CANCEL", "QUIT"}
POKEMON_SUBMENU_KEYWORDS = {"STATS", "STATUS", "SWITCH", "SUMMARY", "CANCEL", "QUIT"}
FIELD_MOVE_KEYWORDS = {"CUT", "FLY", "SURF", "STRENGTH", "FLASH"}


@dataclass(slots=True)
class MenuIntent:
    milestone: Milestone
    required_hm: str | None
    hm_item_name: str | None
    target_map_ready: bool
    should_open_for_hm: bool
    should_teach_hm: bool
    shopping_intent: bool


@dataclass(slots=True)
class MenuSnapshot:
    menu_type: str
    labels: list[str]
    cursor_index: int | None
    top_x: int | None
    top_y: int | None
    max_item: int | None


class MenuManager:
    def __init__(self) -> None:
        self._last_menu_type: str | None = None
        self._runtime: dict[str, CandidateRuntime] = {}

    def reset(self) -> None:
        self._last_menu_type = None
        self._runtime = {}

    def build_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
    ) -> list[CandidateNextStep]:
        self._runtime = {}
        if state.menu_open:
            intent = self._menu_intent(state)
            snapshot = self._detect_menu(state)
            self._last_menu_type = snapshot.menu_type
            return self._build_open_menu_candidates(state, objective_id, snapshot, intent)

        self._last_menu_type = None
        if state.mode.value != "OVERWORLD":
            return []

        intent = self._menu_intent(state)
        if not intent.should_open_for_hm:
            return []

        hm_label = intent.required_hm or "menu target"
        candidate = CandidateNextStep(
            id=f"open_start_menu_for_{self._slugify(hm_label)}",
            type="OPEN_START_MENU_FOR_HM",
            target=ObjectiveTarget(kind="menu", map_id=state.map_id, map_name=state.map_name, detail=hm_label),
            why=f"{hm_label} is required on {intent.milestone.target_map_name}, so open the start menu now.",
            priority=97,
            expected_success_signal="The start menu opens",
            objective_id=objective_id,
        )
        self._runtime[candidate.id] = CandidateRuntime(
            action=ActionDecision(action=ActionType.PRESS_START, repeat=1, reason=f"open start menu for {hm_label}"),
        )
        return [candidate]

    def runtime_map(self) -> dict[str, CandidateRuntime]:
        return dict(self._runtime)

    def _build_open_menu_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        snapshot: MenuSnapshot,
        intent: MenuIntent,
    ) -> list[CandidateNextStep]:
        builders = {
            START_MENU_TYPE: self._start_menu_candidates,
            ITEM_MENU_TYPE: self._item_menu_candidates,
            PARTY_MENU_TYPE: self._party_menu_candidates,
            POKEMON_SUBMENU_TYPE: self._pokemon_submenu_candidates,
            SHOP_MENU_TYPE: self._shop_menu_candidates,
            SHOP_BUY_MENU_TYPE: self._shop_buy_candidates,
            GENERIC_MENU_TYPE: self._generic_menu_candidates,
            UNKNOWN_MENU_TYPE: self._generic_menu_candidates,
        }
        builder = builders.get(snapshot.menu_type, self._generic_menu_candidates)
        candidates = builder(state, objective_id, snapshot, intent)
        if not any(candidate.type == "CLOSE_MENU" for candidate in candidates):
            candidates.append(self._close_menu_candidate(state, objective_id, priority=45, why="Back out if this menu does not help the current milestone."))
        return candidates

    def _start_menu_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        snapshot: MenuSnapshot,
        intent: MenuIntent,
    ) -> list[CandidateNextStep]:
        labels = snapshot.labels or self._default_start_menu_labels(state)
        candidates: list[CandidateNextStep] = []
        for index, label in enumerate(labels):
            normalized = self._normalize_label(label)
            if not normalized:
                continue
            priority = 12
            candidate_type = "SELECT_START_MENU_OPTION"
            why = f"Navigate the start menu to {label}."
            if normalized == "ITEM" and intent.should_teach_hm:
                priority = 96
                candidate_type = "SELECT_START_MENU_OPTION_FOR_HM"
                why = f"Teach {intent.required_hm} by opening the bag and selecting {intent.hm_item_name}."
            elif normalized == "POKEMON" and intent.required_hm:
                priority = 91 if intent.should_teach_hm else 96
                candidate_type = "SELECT_START_MENU_OPTION_FOR_HM"
                why = f"Use the Pokemon menu to find a party member that can use {intent.required_hm}."
            elif normalized == "ITEM" and state.inventory:
                priority = 42
                why = "Use the bag when items are relevant."
            elif normalized == "POKEDEX":
                priority = 6
            elif normalized in {"SAVE", "OPTION"}:
                priority = 8
            elif normalized == "EXIT":
                priority = 18
            candidates.append(
                self._menu_option_candidate(
                    state,
                    objective_id,
                    snapshot,
                    target_index=index,
                    label=label,
                    priority=priority,
                    why=why,
                    candidate_type=candidate_type,
                )
            )
        candidates.append(self._close_menu_candidate(state, objective_id, priority=35, why="Close the start menu if no stronger menu goal is active."))
        return candidates

    def _item_menu_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        snapshot: MenuSnapshot,
        intent: MenuIntent,
    ) -> list[CandidateNextStep]:
        candidates: list[CandidateNextStep] = []
        item_names = [item.name for item in state.inventory]
        visible_labels = snapshot.labels or item_names[: min(len(item_names), 6)]

        if intent.should_teach_hm and intent.hm_item_name:
            target_index = self._find_visible_target_index(visible_labels, intent.hm_item_name)
            if target_index is not None:
                candidates.append(
                    self._menu_option_candidate(
                        state,
                        objective_id,
                        snapshot,
                        target_index=target_index,
                        label=intent.hm_item_name,
                        priority=98,
                        why=f"{intent.hm_item_name} is in the bag and teaching {intent.required_hm} is the current blocker.",
                        candidate_type="SELECT_HM_ITEM",
                        success_signal="The HM is chosen or the party selection menu opens",
                    )
                )

        if visible_labels:
            current_index = self._coerce_index(snapshot.cursor_index, len(visible_labels))
            if current_index is not None:
                current_label = visible_labels[current_index]
                candidates.append(
                    self._menu_option_candidate(
                        state,
                        objective_id,
                        snapshot,
                        target_index=current_index,
                        label=current_label,
                        priority=32,
                        why=f"Use the highlighted bag item {current_label} if it helps the milestone.",
                        candidate_type="SELECT_ITEM_MENU_OPTION",
                        success_signal="The selected item is used or a follow-up menu opens",
                    )
                )

        candidates.append(self._close_menu_candidate(state, objective_id, priority=30, why="Back out of the bag if it is not the right time to use an item."))
        return candidates

    def _party_menu_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        snapshot: MenuSnapshot,
        intent: MenuIntent,
    ) -> list[CandidateNextStep]:
        candidates: list[CandidateNextStep] = []
        party_labels = snapshot.labels or [member.name for member in state.party]
        for index, member in enumerate(state.party[: len(party_labels)]):
            priority = 24
            if intent.required_hm:
                hp = member.hp or 0
                max_hp = member.max_hp or max(hp, 1)
                hp_ratio = 0.0 if max_hp <= 0 else hp / max_hp
                priority = 88 - index * 3
                if hp <= 0:
                    priority -= 24
                elif hp_ratio >= 0.75:
                    priority += 6
                if member.status:
                    priority -= 4
            candidates.append(
                self._menu_option_candidate(
                    state,
                    objective_id,
                    snapshot,
                    target_index=index,
                    label=party_labels[index],
                    priority=priority,
                    why=(
                        f"Select {member.name} to continue the {intent.required_hm} menu flow."
                        if intent.required_hm
                        else f"Select {member.name} from the party menu."
                    ),
                    candidate_type="SELECT_PARTY_POKEMON",
                    success_signal="The selected Pokemon submenu opens or the party state changes",
                )
            )
        candidates.append(self._close_menu_candidate(state, objective_id, priority=34, why="Back out of the party menu if no party action is needed."))
        return candidates

    def _pokemon_submenu_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        snapshot: MenuSnapshot,
        intent: MenuIntent,
    ) -> list[CandidateNextStep]:
        candidates: list[CandidateNextStep] = []
        labels = snapshot.labels
        hm_target = None
        if intent.required_hm:
            hm_target = self._find_visible_target_index(labels, intent.required_hm)
        if hm_target is not None:
            candidates.append(
                self._menu_option_candidate(
                    state,
                    objective_id,
                    snapshot,
                    target_index=hm_target,
                    label=labels[hm_target],
                    priority=100,
                    why=f"Use {intent.required_hm} from the Pokemon submenu.",
                    candidate_type="USE_FIELD_MOVE",
                    success_signal=f"{intent.required_hm} is used or the overworld changes",
                )
            )
        elif intent.required_hm:
            candidates.append(self._close_menu_candidate(state, objective_id, priority=82, why=f"Back out and try a different Pokemon if {intent.required_hm} is not visible here."))

        for index, label in enumerate(labels):
            normalized = self._normalize_label(label)
            if normalized in FIELD_MOVE_KEYWORDS and hm_target is None:
                candidates.append(
                    self._menu_option_candidate(
                        state,
                        objective_id,
                        snapshot,
                        target_index=index,
                        label=label,
                        priority=68,
                        why=f"Use the visible field move {label}.",
                        candidate_type="USE_FIELD_MOVE",
                        success_signal=f"{label} is used or the overworld changes",
                    )
                )
                break
        return candidates

    def _shop_menu_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        snapshot: MenuSnapshot,
        intent: MenuIntent,
    ) -> list[CandidateNextStep]:
        candidates: list[CandidateNextStep] = []
        for index, label in enumerate(snapshot.labels):
            normalized = self._normalize_label(label)
            priority = 16
            if normalized == "BUY":
                priority = 74 if intent.shopping_intent else 28
            elif normalized == "SELL":
                priority = 18
            elif normalized in {"CANCEL", "QUIT"}:
                priority = 34
            candidates.append(
                self._menu_option_candidate(
                    state,
                    objective_id,
                    snapshot,
                    target_index=index,
                    label=label,
                    priority=priority,
                    why=f"Choose {label} in the shop menu.",
                    candidate_type="SELECT_SHOP_MENU_OPTION",
                    success_signal=f"{label} is confirmed or the shop menu changes",
                )
            )
        return candidates

    def _shop_buy_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        snapshot: MenuSnapshot,
        intent: MenuIntent,
    ) -> list[CandidateNextStep]:
        candidates: list[CandidateNextStep] = []
        for index, label in enumerate(snapshot.labels[:4]):
            if not self._normalize_label(label):
                continue
            candidates.append(
                self._menu_option_candidate(
                    state,
                    objective_id,
                    snapshot,
                    target_index=index,
                    label=label,
                    priority=70 - index * 3 if intent.shopping_intent else 20 - index,
                    why=f"Buy {label} from the visible shop list.",
                    candidate_type="BUY_SHOP_ITEM",
                    success_signal=f"{label} is bought or the money/item count changes",
                )
            )
        return candidates

    def _generic_menu_candidates(
        self,
        state: StructuredGameState,
        objective_id: str,
        snapshot: MenuSnapshot,
        intent: MenuIntent,
    ) -> list[CandidateNextStep]:
        candidates: list[CandidateNextStep] = []
        for index, label in enumerate(snapshot.labels[:4]):
            if not self._normalize_label(label):
                continue
            candidates.append(
                self._menu_option_candidate(
                    state,
                    objective_id,
                    snapshot,
                    target_index=index,
                    label=label,
                    priority=20 - index,
                    why=f"Select {label} from the current menu.",
                    candidate_type="SELECT_GENERIC_MENU_OPTION",
                    success_signal=f"{label} is confirmed or the menu state changes",
                )
            )
        return candidates

    def _menu_option_candidate(
        self,
        state: StructuredGameState,
        objective_id: str,
        snapshot: MenuSnapshot,
        *,
        target_index: int,
        label: str,
        priority: int,
        why: str,
        candidate_type: str,
        success_signal: str | None = None,
    ) -> CandidateNextStep:
        action, _step_budget = self._menu_action(snapshot.cursor_index, target_index)
        slug = self._slugify(label) or f"item_{target_index}"
        candidate = CandidateNextStep(
            id=f"{candidate_type.lower()}_{slug}",
            type=candidate_type,
            target=ObjectiveTarget(kind="menu", map_id=state.map_id, map_name=state.map_name, detail=label),
            why=why,
            priority=priority,
            expected_success_signal=success_signal or f"The cursor moves toward {label} or {label} is confirmed",
            objective_id=objective_id,
        )
        self._runtime[candidate.id] = CandidateRuntime(
            action=ActionDecision(action=action, repeat=1, reason=why),
        )
        return candidate

    def _close_menu_candidate(
        self,
        state: StructuredGameState,
        objective_id: str,
        *,
        priority: int,
        why: str,
    ) -> CandidateNextStep:
        candidate = CandidateNextStep(
            id="close_menu",
            type="CLOSE_MENU",
            target=ObjectiveTarget(kind="menu", map_id=state.map_id, map_name=state.map_name, detail="close"),
            why=why,
            priority=priority,
            expected_success_signal="The menu closes or backs out one level",
            objective_id=objective_id,
        )
        self._runtime[candidate.id] = CandidateRuntime(
            action=ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="close menu"),
        )
        return candidate

    def _detect_menu(self, state: StructuredGameState) -> MenuSnapshot:
        ui = self._ui_context(state)
        top_x = self._as_int(ui.get("top_menu_item_x"))
        top_y = self._as_int(ui.get("top_menu_item_y"))
        cursor_index = self._as_int(ui.get("current_menu_item"))
        max_item = self._as_int(ui.get("max_menu_item"))
        labels = self._decode_menu_labels(state.game_area, top_x, top_y, max_item)

        normalized_labels = {self._normalize_label(label) for label in labels if self._normalize_label(label)}
        menu_type = UNKNOWN_MENU_TYPE
        if normalized_labels & START_MENU_KEYWORDS:
            menu_type = START_MENU_TYPE
        elif normalized_labels & FIELD_MOVE_KEYWORDS or normalized_labels & POKEMON_SUBMENU_KEYWORDS:
            menu_type = POKEMON_SUBMENU_TYPE
        elif normalized_labels & SHOP_MENU_KEYWORDS:
            menu_type = SHOP_MENU_TYPE
        elif self._labels_match_inventory(labels, state):
            menu_type = ITEM_MENU_TYPE
        elif self._labels_match_party(labels, state):
            menu_type = PARTY_MENU_TYPE
        elif labels and ("mart" in state.map_name.lower() or self._last_menu_type == SHOP_MENU_TYPE):
            menu_type = SHOP_BUY_MENU_TYPE
        elif labels:
            menu_type = GENERIC_MENU_TYPE
        else:
            menu_type = self._fallback_menu_type(state, top_x, max_item)

        return MenuSnapshot(
            menu_type=menu_type,
            labels=labels,
            cursor_index=cursor_index,
            top_x=top_x,
            top_y=top_y,
            max_item=max_item,
        )

    def _menu_intent(self, state: StructuredGameState) -> MenuIntent:
        milestone = get_current_milestone(
            state.story_flags,
            [item.name for item in state.inventory],
            current_map_name=state.map_name,
            badges=state.badges,
        )
        required_hm = milestone.required_hms[0] if milestone.required_hms else None
        hm_item_name = None if required_hm is None else f"HM {required_hm}"
        target_map_ready = self._map_matches(state.map_name, milestone.target_map_name)
        should_teach_hm = bool(
            required_hm
            and hm_item_name in {item.name for item in state.inventory}
            and target_map_ready
            and self._milestone_mentions(milestone, f"teach {required_hm}")
        )
        return MenuIntent(
            milestone=milestone,
            required_hm=required_hm,
            hm_item_name=hm_item_name,
            target_map_ready=target_map_ready,
            should_open_for_hm=bool(required_hm and target_map_ready),
            should_teach_hm=should_teach_hm,
            shopping_intent=self._milestone_mentions(milestone, "shop"),
        )

    def _decode_menu_labels(
        self,
        game_area: list[list[int]] | None,
        top_x: int | None,
        top_y: int | None,
        max_item: int | None,
    ) -> list[str]:
        if game_area is None or top_x is None or top_y is None or max_item is None:
            return []
        if max_item < 0:
            return []
        labels: list[str] = []
        for index in range(min(max_item + 1, 8)):
            y = top_y + (index * 2)
            label = self._decode_best_menu_row(game_area, y, top_x + 1)
            labels.append(label)
        while labels and not labels[-1]:
            labels.pop()
        return labels

    def _decode_best_menu_row(
        self,
        game_area: list[list[int]],
        y: int,
        x: int,
    ) -> str:
        if not game_area:
            return ""
        height = len(game_area)
        width = len(game_area[0]) if game_area[0] else 0
        if height == 0 or width == 0:
            return ""

        candidates: list[str] = []
        for row in (y, y + 1):
            if row < 0 or row >= height:
                continue
            for col in (x, x + 1):
                if col < 0 or col >= width:
                    continue
                decoded = "".join(POKEMON_RED_CHAR_MAP.get(int(tile), " ") for tile in game_area[row][col:width])
                candidates.append(self._compress_text(decoded))
        if not candidates:
            return ""
        return max(candidates, key=self._readable_text_score)

    def _labels_match_inventory(self, labels: list[str], state: StructuredGameState) -> bool:
        inventory_names = [self._normalize_lookup(item.name) for item in state.inventory]
        if not labels or not inventory_names:
            return False
        matches = 0
        for label in labels:
            normalized = self._normalize_lookup(label)
            if any(normalized.startswith(name) or name in normalized for name in inventory_names if name):
                matches += 1
        return matches >= 1

    def _labels_match_party(self, labels: list[str], state: StructuredGameState) -> bool:
        party_names = [self._normalize_lookup(member.name) for member in state.party]
        if not labels or not party_names:
            return False
        matches = 0
        for label in labels:
            normalized = self._normalize_lookup(label)
            if any(normalized.startswith(name) or name in normalized for name in party_names if name):
                matches += 1
        return matches >= 1

    def _fallback_menu_type(
        self,
        state: StructuredGameState,
        top_x: int | None,
        max_item: int | None,
    ) -> str:
        if top_x is None or max_item is None:
            return UNKNOWN_MENU_TYPE
        if top_x >= 10 and max_item >= 3:
            return START_MENU_TYPE
        if top_x >= 10 and max_item <= 3 and self._last_menu_type == PARTY_MENU_TYPE:
            return POKEMON_SUBMENU_TYPE
        if top_x <= 2:
            party_delta = abs(max_item - max(len(state.party) - 1, 0)) if state.party else 999
            item_delta = abs(max_item - max(len(state.inventory) - 1, 0)) if state.inventory else 999
            if party_delta <= item_delta:
                return PARTY_MENU_TYPE if state.party else UNKNOWN_MENU_TYPE
            return ITEM_MENU_TYPE if state.inventory else UNKNOWN_MENU_TYPE
        if self._last_menu_type == SHOP_MENU_TYPE and state.map_name and "mart" in state.map_name.lower():
            return SHOP_BUY_MENU_TYPE
        return UNKNOWN_MENU_TYPE

    def _default_start_menu_labels(self, state: StructuredGameState) -> list[str]:
        labels = ["POKEMON", "ITEM", "SAVE", "EXIT"]
        if "got_pokedex" in state.story_flags:
            labels.insert(0, "POKEDEX")
        return labels

    def _menu_action(
        self,
        cursor_index: int | None,
        target_index: int,
    ) -> tuple[ActionType, int]:
        if cursor_index is None:
            if target_index <= 0:
                return ActionType.PRESS_A, 2
            return ActionType.MOVE_DOWN, target_index + 1
        if cursor_index == target_index:
            return ActionType.PRESS_A, 1
        if cursor_index < target_index:
            return ActionType.MOVE_DOWN, (target_index - cursor_index) + 1
        return ActionType.MOVE_UP, (cursor_index - target_index) + 1

    def _coerce_index(self, cursor_index: int | None, length: int) -> int | None:
        if cursor_index is None or length <= 0:
            return None
        if cursor_index < 0:
            return None
        return min(cursor_index, length - 1)

    def _find_visible_target_index(self, labels: list[str], target: str) -> int | None:
        if not labels:
            return None
        target_normalized = self._normalize_lookup(target)
        for index, label in enumerate(labels):
            normalized = self._normalize_lookup(label)
            if normalized == target_normalized or normalized.startswith(target_normalized) or target_normalized in normalized:
                return index
        return None

    def _milestone_mentions(self, milestone: Milestone, text: str) -> bool:
        needle = self._normalize_lookup(text)
        haystacks = [milestone.description, *milestone.route_hints, *milestone.sub_steps]
        return any(needle in self._normalize_lookup(value) for value in haystacks)

    def _map_matches(self, current_map_name: str | None, target_map_name: str) -> bool:
        if not current_map_name:
            return False
        current_tokens = self._tokenize_name(current_map_name)
        target_tokens = self._tokenize_name(target_map_name)
        if not current_tokens or not target_tokens:
            return False
        if target_tokens.issubset(current_tokens) or current_tokens.issubset(target_tokens):
            return True
        generic_tokens = {
            "city",
            "town",
            "route",
            "road",
            "house",
            "gym",
            "lab",
            "forest",
            "cave",
            "tower",
            "dock",
            "center",
            "pokecenter",
            "room",
            "rooms",
            "island",
            "plateau",
            "gate",
            "hideout",
            "mansion",
            "co",
            "lobby",
            "floor",
        }
        return bool((current_tokens - generic_tokens) & (target_tokens - generic_tokens))

    def _ui_context(self, state: StructuredGameState) -> dict[str, object]:
        ram_context = state.metadata.get("ram_context")
        if isinstance(ram_context, dict):
            ui = ram_context.get("ui")
            if isinstance(ui, dict):
                return ui
        return {}

    def _as_int(self, value: object) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _compress_text(self, value: str) -> str:
        compact = re.sub(r"\s+", " ", value).strip()
        compact = compact.replace(" '", "'")
        compact = compact.replace('" ', '"')
        return compact

    def _readable_text_score(self, value: str) -> tuple[int, int]:
        alpha = sum(char.isalnum() for char in value)
        return (alpha, len(value))

    def _normalize_label(self, value: str) -> str:
        return self._compress_text(value).upper()

    def _normalize_lookup(self, value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    def _tokenize_name(self, value: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token}

    def _slugify(self, value: str) -> str:
        return "_".join(re.findall(r"[a-z0-9]+", value.lower()))
