from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
import json
import re
from typing import Any

from pokemon_agent.agent.context_manager import measure_prompt
from pokemon_agent.agent.game_knowledge import GameKnowledge
from pokemon_agent.agent.game_knowledge import load_game_knowledge
from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.llm_client import CompletionResponse
from pokemon_agent.agent.llm_client import LLMUsage
from pokemon_agent.agent.planning_types import PlanningResult
from pokemon_agent.data.pokemon_red_battle_data import POKE_BALL_ITEM_NAMES
from pokemon_agent.data.pokemon_red_battle_data import POTION_ITEM_NAMES
from pokemon_agent.data.pokemon_red_battle_data import SPECIES_TYPE_MAP
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import BattleContext
from pokemon_agent.models.state import MoveInfo
from pokemon_agent.models.state import StructuredGameState

MAIN_MENU_POSITIONS = {
    "FIGHT": 0,
    "POKEMON": 1,
    "BAG": 2,
    "RUN": 3,
}
MAIN_MENU_COORDS = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1),
}
IMPORTANT_BATTLE_MAP_TOKENS = ("Gym", "Lorelei", "Bruno", "Agatha", "Lance", "Champion")
COMMON_WILD_SPECIES = {
    "Caterpie",
    "Geodude",
    "Magikarp",
    "Metapod",
    "Oddish",
    "Paras",
    "Pidgey",
    "Rattata",
    "Spearow",
    "Tentacool",
    "Venonat",
    "Weedle",
    "Zubat",
}
_BATTLE_STRATEGY_SYSTEM_PROMPT = (
    "You are a Pokemon Red battle strategist. "
    "Return exactly one JSON object with the requested keys and no markdown."
)


@dataclass(frozen=True, slots=True)
class BattleStrategy:
    lead_pokemon: str | None = None
    preferred_moves: list[str] = field(default_factory=list)
    switch_threshold_hp_pct: int = 25
    switch_target: str | None = None
    use_items: bool = True
    should_catch: bool = False
    should_run: bool = False
    notes: str = ""


@dataclass(frozen=True, slots=True)
class _BattleTarget:
    kind: str
    move_index: int | None = None
    item_index: int | None = None
    item_name: str | None = None
    party_index: int | None = None
    pokemon_name: str | None = None


@dataclass(slots=True)
class _StrategyResult:
    strategy: BattleStrategy
    llm_attempted: bool = False
    used_fallback: bool = False
    raw_response: str | None = None
    llm_usage: LLMUsage | None = None
    llm_model: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    prompt_metrics: dict[str, Any] | None = None


BattleCompletion = Callable[[list[dict[str, str]], str], CompletionResponse]


class BattleController:
    def __init__(self, complete: BattleCompletion | None = None, knowledge: GameKnowledge | None = None) -> None:
        self._complete = complete
        self._knowledge = knowledge or load_game_knowledge()
        self._strategy: BattleStrategy | None = None
        self._strategy_key: tuple[str | None, str | None, int | None] | None = None
        self._submenu = "MAIN"

    def step(self, state: StructuredGameState, context: TurnContext) -> PlanningResult:
        battle = state.battle_state
        if battle is None:
            self.reset()
            return PlanningResult(
                action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="advance battle transition"),
                planner_source="battle_controller",
            )

        self._sync_submenu(battle)
        planning = PlanningResult(planner_source="battle_controller")
        if self._needs_new_strategy(battle):
            strategy_result = self._build_strategy(state, battle, context)
            self._strategy = strategy_result.strategy
            self._strategy_key = self._strategy_identity(battle)
            planning.used_fallback = strategy_result.used_fallback
            planning.raw_response = strategy_result.raw_response
            planning.llm_attempted = strategy_result.llm_attempted
            planning.llm_usage = strategy_result.llm_usage
            planning.llm_model = strategy_result.llm_model
            planning.messages = list(strategy_result.messages)
            planning.prompt_metrics = strategy_result.prompt_metrics

        strategy = self._strategy or self._heuristic_strategy(state, battle)
        target = self._resolve_target(state, battle, strategy)
        planning.action = self._action_for_target(battle, target)
        return planning

    def reset(self) -> None:
        self._strategy = None
        self._strategy_key = None
        self._submenu = "MAIN"

    def _needs_new_strategy(self, battle: BattleContext) -> bool:
        return self._strategy is None or self._strategy_key != self._strategy_identity(battle)

    def _strategy_identity(self, battle: BattleContext) -> tuple[str | None, str | None, int | None]:
        return (battle.kind, battle.enemy_species, battle.enemy_level)

    def _build_strategy(
        self,
        state: StructuredGameState,
        battle: BattleContext,
        context: TurnContext,
    ) -> _StrategyResult:
        heuristic = self._heuristic_strategy(state, battle)
        if self._complete is None or self._is_trivial_wild(state, battle):
            return _StrategyResult(strategy=heuristic)

        messages = self._strategy_messages(state, battle, context, heuristic)
        prompt_metrics = measure_prompt(messages)
        try:
            response = self._complete(messages, "battle strategy")
            strategy = self._parse_strategy(response.content, state, battle, heuristic)
            return _StrategyResult(
                strategy=strategy,
                llm_attempted=True,
                raw_response=response.content,
                llm_usage=response.usage,
                llm_model=response.model,
                messages=messages,
                prompt_metrics=prompt_metrics,
            )
        except Exception:
            return _StrategyResult(
                strategy=heuristic,
                llm_attempted=True,
                used_fallback=True,
                messages=messages,
                prompt_metrics=prompt_metrics,
            )

    def _strategy_messages(
        self,
        state: StructuredGameState,
        battle: BattleContext,
        context: TurnContext,
        heuristic: BattleStrategy,
    ) -> list[dict[str, str]]:
        payload = {
            "response_schema": {
                "lead_pokemon": "<string|null>",
                "preferred_moves": ["<move name>", "<move name>"],
                "switch_threshold_hp_pct": "<integer 0-100>",
                "switch_target": "<string|null>",
                "use_items": "<boolean>",
                "should_catch": "<boolean>",
                "should_run": "<boolean>",
                "notes": "<short string>",
            },
            "context": {
                "map_name": state.map_name,
                "objective_id": context.objective.id if context.objective is not None else None,
                "battle": battle.model_dump(mode="json"),
                "party": [self._party_summary(member.model_dump(mode="json")) for member in state.party],
                "inventory": [item.model_dump(mode="json") for item in state.inventory if item.count > 0],
                "heuristic_strategy": {
                    "lead_pokemon": heuristic.lead_pokemon,
                    "preferred_moves": list(heuristic.preferred_moves),
                    "switch_threshold_hp_pct": heuristic.switch_threshold_hp_pct,
                    "switch_target": heuristic.switch_target,
                    "use_items": heuristic.use_items,
                    "should_catch": heuristic.should_catch,
                    "should_run": heuristic.should_run,
                    "notes": heuristic.notes,
                },
                "battle_tags": {
                    "trivial_wild": self._is_trivial_wild(state, battle),
                    "important_battle": self._is_important_battle(state, battle),
                    "catch_interest": self._has_catch_interest(state, battle),
                },
            },
        }
        return [
            {"role": "system", "content": _BATTLE_STRATEGY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
        ]

    def _party_summary(self, member: dict[str, Any]) -> dict[str, Any]:
        species = member.get("name")
        types = self._species_types(species)
        return {
            **member,
            "types": [battle_type for battle_type in types or () if battle_type],
        }

    def _parse_strategy(
        self,
        raw_text: str,
        state: StructuredGameState,
        battle: BattleContext,
        heuristic: BattleStrategy,
    ) -> BattleStrategy:
        payload = json.loads(self._extract_json(raw_text))
        preferred_moves = self._normalize_preferred_moves(payload.get("preferred_moves"), battle, heuristic.preferred_moves)
        lead = self._resolve_party_name(state, payload.get("lead_pokemon"))
        switch_target = self._resolve_party_name(state, payload.get("switch_target"))
        strategy = BattleStrategy(
            lead_pokemon=lead,
            preferred_moves=preferred_moves or list(heuristic.preferred_moves),
            switch_threshold_hp_pct=self._clamp_int(
                payload.get("switch_threshold_hp_pct"),
                default=heuristic.switch_threshold_hp_pct,
                minimum=0,
                maximum=100,
            ),
            switch_target=switch_target,
            use_items=self._coerce_bool(payload.get("use_items"), heuristic.use_items),
            should_catch=self._coerce_bool(payload.get("should_catch"), heuristic.should_catch),
            should_run=self._coerce_bool(payload.get("should_run"), heuristic.should_run),
            notes=str(payload.get("notes") or "").strip(),
        )
        if battle.kind != "WILD":
            strategy = BattleStrategy(
                lead_pokemon=strategy.lead_pokemon,
                preferred_moves=list(strategy.preferred_moves),
                switch_threshold_hp_pct=strategy.switch_threshold_hp_pct,
                switch_target=strategy.switch_target,
                use_items=strategy.use_items,
                should_catch=False,
                should_run=False,
                notes=strategy.notes,
            )
        return strategy

    def _resolve_target(
        self,
        state: StructuredGameState,
        battle: BattleContext,
        strategy: BattleStrategy,
    ) -> _BattleTarget:
        if battle.player_active_hp is not None and battle.player_active_hp <= 0:
            return self._switch_target(state, battle, strategy.switch_target)

        if battle.kind == "WILD" and strategy.should_run:
            return _BattleTarget(kind="run")

        ball = self._first_inventory_match(state, POKE_BALL_ITEM_NAMES)
        if battle.kind == "WILD" and strategy.should_catch and ball is not None:
            return _BattleTarget(kind="item", item_index=ball[0], item_name=ball[1])

        lead_target = self._party_index_for_name(state, battle.player_active_species, strategy.lead_pokemon)
        if lead_target is not None:
            return _BattleTarget(kind="switch", party_index=lead_target[0], pokemon_name=lead_target[1])

        hp_pct = self._hp_pct(battle.player_active_hp, battle.player_active_max_hp)
        if hp_pct is not None and hp_pct < strategy.switch_threshold_hp_pct:
            potion = self._first_inventory_match(state, POTION_ITEM_NAMES)
            if strategy.use_items and potion is not None:
                return _BattleTarget(kind="item", item_index=potion[0], item_name=potion[1])
            switch_target = self._switch_target(state, battle, strategy.switch_target)
            if switch_target.kind == "switch":
                return switch_target

        preferred_index = self._preferred_move_index(battle, strategy.preferred_moves)
        if preferred_index is not None:
            return _BattleTarget(kind="move", move_index=preferred_index)

        strongest = self._strongest_move_index(battle)
        if strongest is not None:
            return _BattleTarget(kind="move", move_index=strongest)

        fallback = self._first_available_move_index(battle)
        if fallback is not None:
            return _BattleTarget(kind="move", move_index=fallback)

        return _BattleTarget(kind="confirm")

    def _switch_target(
        self,
        state: StructuredGameState,
        battle: BattleContext,
        preferred_name: str | None,
    ) -> _BattleTarget:
        preferred = self._party_index_for_name(state, battle.player_active_species, preferred_name)
        if preferred is not None:
            return _BattleTarget(kind="switch", party_index=preferred[0], pokemon_name=preferred[1])
        fallback = self._best_switch_target(state, battle)
        if fallback is not None:
            return _BattleTarget(kind="switch", party_index=fallback[0], pokemon_name=fallback[1])
        return _BattleTarget(kind="confirm")

    def _action_for_target(self, battle: BattleContext, target: _BattleTarget) -> ActionDecision:
        if target.kind == "move" and target.move_index is not None:
            action = self._action_for_move_selection(battle, target.move_index)
            self._submenu = "FIGHT"
            return action
        if target.kind == "item" and target.item_index is not None:
            action = self._action_for_item_selection(battle, target.item_index)
            self._submenu = "BAG"
            return action
        if target.kind == "switch" and target.party_index is not None:
            action = self._action_for_party_selection(battle, target.party_index)
            self._submenu = "POKEMON"
            return action
        if target.kind == "run":
            action = self._action_for_main_menu_target(battle, MAIN_MENU_POSITIONS["RUN"], "RUN")
            self._submenu = "RUN"
            return action
        return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="advance battle")

    def _sync_submenu(self, battle: BattleContext) -> None:
        if battle.player_active_hp is not None and battle.player_active_hp <= 0:
            self._submenu = "POKEMON"
            return

        move_cursor = self._move_cursor_position(battle)
        if self._submenu == "FIGHT":
            move_slots = max(1, len([move for move in battle.available_moves if move.pp >= 0]))
            if move_cursor is not None and 1 <= move_cursor <= move_slots:
                self._submenu = "FIGHT"
                return

        position = battle.battle_menu_position
        if self._submenu in {"BAG", "POKEMON"} and position is not None and position not in MAIN_MENU_COORDS:
            return
        if position in MAIN_MENU_COORDS:
            self._submenu = "MAIN"

    def _action_for_move_selection(self, battle: BattleContext, move_index: int) -> ActionDecision:
        if self._submenu == "POKEMON":
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="return to battle menu")
        if self._submenu == "BAG":
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="leave bag and return to battle menu")
        if self._submenu != "FIGHT":
            return self._action_for_main_menu_target(battle, MAIN_MENU_POSITIONS["FIGHT"], "FIGHT")

        current = self._move_cursor_position(battle)
        if current is None:
            return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="open or confirm move")
        if current < move_index:
            return ActionDecision(action=ActionType.MOVE_DOWN, repeat=1, reason="move to the next move")
        if current > move_index:
            return ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="move to the selected move")
        return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="confirm move")

    def _action_for_item_selection(self, battle: BattleContext, item_index: int) -> ActionDecision:
        if self._submenu == "FIGHT":
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="leave move list")
        if self._submenu != "BAG":
            return self._action_for_main_menu_target(battle, MAIN_MENU_POSITIONS["BAG"], "BAG")

        current = max(0, battle.battle_menu_position or 0)
        if current < item_index:
            return ActionDecision(action=ActionType.MOVE_DOWN, repeat=1, reason="move to the bag item")
        if current > item_index:
            return ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="move to the bag item")
        return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="use the highlighted item")

    def _action_for_party_selection(self, battle: BattleContext, target_index: int) -> ActionDecision:
        if self._submenu == "FIGHT":
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="leave move list")
        if self._submenu == "BAG":
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="leave bag")
        if self._submenu != "POKEMON":
            return self._action_for_main_menu_target(battle, MAIN_MENU_POSITIONS["POKEMON"], "POKEMON")

        current = max(0, battle.battle_menu_position or 0)
        if current < target_index:
            return ActionDecision(action=ActionType.MOVE_DOWN, repeat=1, reason="move to the next Pokemon")
        if current > target_index:
            return ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="move to the selected Pokemon")
        return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="confirm switch")

    def _action_for_main_menu_target(self, battle: BattleContext, target: int, label: str) -> ActionDecision:
        current = battle.battle_menu_position
        if current is None:
            return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason=f"advance battle toward {label.lower()}")
        if current not in MAIN_MENU_COORDS:
            if self._submenu == "MAIN":
                return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason=f"advance battle toward {label.lower()}")
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason=f"return to the {label.lower()} target")
        if current == target:
            return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason=f"open {label.lower()}")

        current_x, current_y = MAIN_MENU_COORDS[current]
        target_x, target_y = MAIN_MENU_COORDS[target]
        if current_x < target_x:
            return ActionDecision(action=ActionType.MOVE_RIGHT, repeat=1, reason=f"move toward {label.lower()}")
        if current_x > target_x:
            return ActionDecision(action=ActionType.MOVE_LEFT, repeat=1, reason=f"move toward {label.lower()}")
        if current_y < target_y:
            return ActionDecision(action=ActionType.MOVE_DOWN, repeat=1, reason=f"move toward {label.lower()}")
        return ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason=f"move toward {label.lower()}")

    def _heuristic_strategy(self, state: StructuredGameState, battle: BattleContext) -> BattleStrategy:
        preferred_moves = self._rank_moves(battle)
        hp_pct = self._hp_pct(battle.player_active_hp, battle.player_active_max_hp)
        catch_interest = self._has_catch_interest(state, battle)
        important_battle = self._is_important_battle(state, battle)
        switch_target = self._best_switch_target(state, battle)
        trivial_wild = self._is_trivial_wild(state, battle)
        should_run = battle.kind == "WILD" and trivial_wild and (hp_pct is None or hp_pct >= 70)
        return BattleStrategy(
            preferred_moves=preferred_moves,
            switch_threshold_hp_pct=20 if trivial_wild else 25,
            switch_target=switch_target[1] if switch_target is not None else None,
            use_items=bool(self._first_inventory_match(state, POTION_ITEM_NAMES)) and not trivial_wild,
            should_catch=catch_interest,
            should_run=should_run,
            notes="heuristic battle strategy" if not important_battle else "heuristic strategy for important battle",
        )

    def _rank_moves(self, battle: BattleContext) -> list[str]:
        player_types = self._species_types(battle.player_active_species)
        enemy_types = self._species_types(battle.enemy_species)
        scored: list[tuple[float, str]] = []
        for move in battle.available_moves:
            if move.pp <= 0:
                continue
            score = self._move_score(move, player_types, enemy_types)
            scored.append((score, move.name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in scored]

    def _preferred_move_index(self, battle: BattleContext, preferred_moves: list[str]) -> int | None:
        normalized_to_index = {
            self._normalize_name(move.name): index
            for index, move in enumerate(battle.available_moves, start=1)
            if move.pp > 0
        }
        for move_name in preferred_moves:
            index = normalized_to_index.get(self._normalize_name(move_name))
            if index is not None:
                return index
        return None

    def _first_available_move_index(self, battle: BattleContext) -> int | None:
        for index, move in enumerate(battle.available_moves, start=1):
            if move.pp > 0:
                return index
        return None

    def _strongest_move_index(self, battle: BattleContext) -> int | None:
        best_index: int | None = None
        best_score = float("-inf")
        player_types = self._species_types(battle.player_active_species)
        enemy_types = self._species_types(battle.enemy_species)
        for index, move in enumerate(battle.available_moves, start=1):
            if move.pp <= 0:
                continue
            score = self._move_score(move, player_types, enemy_types)
            if score > best_score:
                best_score = score
                best_index = index
        return best_index

    def _move_score(
        self,
        move: MoveInfo,
        player_types: tuple[str, str] | None,
        enemy_types: tuple[str, str] | None,
    ) -> float:
        base_power = 45 if move.power == 1 else float(move.power or 0)
        effectiveness = self._type_multiplier(move.move_type, enemy_types)
        stab = 1.5 if player_types and move.move_type in player_types else 1.0
        return base_power * effectiveness * stab

    def _best_switch_target(self, state: StructuredGameState, battle: BattleContext) -> tuple[int, str] | None:
        enemy_types = self._species_types(battle.enemy_species)
        candidates: list[tuple[float, int, str]] = []
        for index, member in enumerate(state.party):
            if not member.hp or member.hp <= 0:
                continue
            if self._normalize_name(member.name) == self._normalize_name(battle.player_active_species):
                continue
            member_types = self._species_types(member.name)
            risk = self._incoming_risk(enemy_types, member_types) if enemy_types and member_types else 1.0
            hp_ratio = 0.0 if member.max_hp in {None, 0} else float(member.hp or 0) / float(member.max_hp or 1)
            candidates.append((risk - hp_ratio, index, member.name))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, index, name = candidates[0]
        return index, name

    def _incoming_risk(
        self,
        attacker_types: tuple[str, str] | None,
        defender_types: tuple[str, str] | None,
    ) -> float:
        if attacker_types is None or defender_types is None:
            return 1.0
        best = 1.0
        for move_type in set(attacker_types):
            best = max(best, self._type_multiplier(move_type, defender_types) * 1.5)
        return best

    def _type_multiplier(self, move_type: str | None, defender_types: tuple[str, str] | None) -> float:
        if move_type is None or defender_types is None:
            return 1.0
        multiplier = 1.0
        for defender_type in set(defender_types):
            multiplier *= self._knowledge.type_effectiveness(move_type, defender_type)
        return multiplier

    def _species_types(self, species_name: str | None) -> tuple[str, str] | None:
        if species_name is None:
            return None
        symbol = species_name.upper().replace(" ", "_").replace(".", "").replace("'", "")
        symbol = symbol.replace("MR__MIME", "MR_MIME")
        symbol = symbol.replace("FARFETCHD", "FARFETCHD")
        symbol = symbol.replace("NIDORAN_M", "NIDORAN_M")
        symbol = symbol.replace("NIDORAN_F", "NIDORAN_F")
        return SPECIES_TYPE_MAP.get(symbol)

    def _first_inventory_match(
        self,
        state: StructuredGameState,
        allowed_names: frozenset[str],
    ) -> tuple[int, str] | None:
        for index, item in enumerate(state.inventory):
            if item.name in allowed_names and item.count > 0:
                return index, item.name
        return None

    def _has_catch_interest(self, state: StructuredGameState, battle: BattleContext) -> bool:
        if battle.kind != "WILD" or not battle.enemy_species:
            return False
        if self._first_inventory_match(state, POKE_BALL_ITEM_NAMES) is None:
            return False
        party_species = {self._normalize_name(member.name) for member in state.party}
        return self._normalize_name(battle.enemy_species) not in party_species

    def _is_trivial_wild(self, state: StructuredGameState, battle: BattleContext) -> bool:
        if battle.kind != "WILD" or self._has_catch_interest(state, battle):
            return False
        levels = [member.level for member in state.party if member.level is not None]
        average_level = round(sum(levels) / len(levels)) if levels else battle.player_active_level
        if battle.enemy_level is not None and average_level is not None and battle.enemy_level <= average_level - 5:
            return True
        if (
            battle.enemy_level is not None
            and battle.player_active_level is not None
            and battle.player_active_level - battle.enemy_level >= 5
            and battle.enemy_species in COMMON_WILD_SPECIES
        ):
            return True
        return False

    def _is_important_battle(self, state: StructuredGameState, battle: BattleContext) -> bool:
        if battle.kind == "TRAINER":
            return True
        return any(token in state.map_name for token in IMPORTANT_BATTLE_MAP_TOKENS)

    def _party_index_for_name(
        self,
        state: StructuredGameState,
        active_species: str | None,
        target_name: str | None,
    ) -> tuple[int, str] | None:
        if not target_name:
            return None
        normalized_target = self._normalize_name(target_name)
        normalized_active = self._normalize_name(active_species)
        for index, member in enumerate(state.party):
            if self._normalize_name(member.name) != normalized_target:
                continue
            if not member.hp or member.hp <= 0:
                return None
            if normalized_target == normalized_active:
                return None
            return index, member.name
        return None

    def _resolve_party_name(self, state: StructuredGameState, value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        normalized = self._normalize_name(value)
        for member in state.party:
            if self._normalize_name(member.name) == normalized:
                return member.name
        return None

    def _normalize_preferred_moves(
        self,
        raw_value: object,
        battle: BattleContext,
        fallback: list[str],
    ) -> list[str]:
        available = {
            self._normalize_name(move.name): move.name
            for move in battle.available_moves
            if move.pp > 0
        }
        preferred: list[str] = []
        if isinstance(raw_value, list):
            for item in raw_value:
                if not isinstance(item, str) or not item.strip():
                    continue
                match = available.get(self._normalize_name(item))
                if match is not None and match not in preferred:
                    preferred.append(match)
        for move_name in fallback:
            match = available.get(self._normalize_name(move_name))
            if match is not None and match not in preferred:
                preferred.append(match)
        return preferred

    def _coerce_bool(self, value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
        return default

    def _clamp_int(self, value: object, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    def _hp_pct(self, hp: int | None, max_hp: int | None) -> int | None:
        if hp is None or max_hp is None or max_hp <= 0:
            return None
        return round((hp / max_hp) * 100)

    def _move_cursor_position(self, battle: BattleContext) -> int | None:
        position = battle.move_cursor_position
        if position is None or position <= 0:
            position = battle.battle_menu_position
        if position is None or position <= 0:
            return None
        return position

    def _extract_json(self, raw_text: str) -> str:
        text = raw_text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("model response did not contain a JSON object")
        return text[start : end + 1]

    def _normalize_name(self, value: str | None) -> str:
        if not value:
            return ""
        return "".join(re.findall(r"[A-Z0-9]+", value.upper()))
