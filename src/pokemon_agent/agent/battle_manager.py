from __future__ import annotations

from dataclasses import dataclass

from pokemon_agent.data.pokemon_red_battle_data import POKE_BALL_ITEM_NAMES
from pokemon_agent.data.pokemon_red_battle_data import POTION_ITEM_NAMES
from pokemon_agent.data.pokemon_red_battle_data import SPECIES_TYPE_MAP
from pokemon_agent.data.pokemon_red_battle_data import TYPE_EFFECTIVENESS
from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.planner import CandidateNextStep
from pokemon_agent.models.planner import CandidateRuntime
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


@dataclass(slots=True)
class BattleTracker:
    battle_key: tuple[str | None, str | None, int | None, str | None] | None = None
    submenu: str = "MAIN"


class BattleManager:
    def __init__(self) -> None:
        self._tracker = BattleTracker()
        self._runtime: dict[str, CandidateRuntime] = {}

    def reset(self) -> None:
        self._tracker = BattleTracker()
        self._runtime = {}

    def build_candidates(self, state: StructuredGameState, objective_id: str | None) -> list[CandidateNextStep]:
        self._runtime = {}
        battle = state.battle_state
        if battle is None:
            self.reset()
            return []

        self._sync_tracker(state, battle)

        if battle.player_active_hp is not None and battle.player_active_hp <= 0:
            switch_candidate = self._build_switch_candidate(state, battle, objective_id, priority=100)
            if switch_candidate is not None:
                return [switch_candidate]
            return [self._fallback_candidate(objective_id, "A Pokemon fainted and battle input is still required.")]

        candidates: list[CandidateNextStep] = []
        hp_ratio = self._hp_ratio(battle.player_active_hp, battle.player_active_max_hp)
        potions = self._indexed_inventory(state, POTION_ITEM_NAMES)
        pokeballs = self._indexed_inventory(state, POKE_BALL_ITEM_NAMES)
        catch_decision = self._needs_catch_decision(state, battle, pokeballs)
        important_trainer = self._needs_llm_strategy(state, battle)
        strongest_move_index = self._strongest_move_index(battle)

        if hp_ratio is not None and hp_ratio < 0.25 and potions:
            potion_candidate = self._build_item_candidate(
                state=state,
                battle=battle,
                objective_id=objective_id,
                item_index=potions[0][0],
                item_name=potions[0][1],
                candidate_id="battle_use_potion",
                candidate_type="USE_POTION",
                why="HP is critical and a healing item is available.",
                priority=100,
            )
            if potion_candidate is not None:
                candidates.append(potion_candidate)

        if self._has_type_disadvantage(state, battle):
            switch_candidate = self._build_switch_candidate(state, battle, objective_id, priority=88 if important_trainer else 72)
            if switch_candidate is not None:
                candidates.append(switch_candidate)

        for index, move in enumerate(battle.available_moves, start=1):
            if move.pp <= 0:
                continue
            priority = self._move_priority(state, battle, move, index, strongest_move_index, important_trainer, catch_decision)
            candidate = CandidateNextStep(
                id=f"battle_fight_move_{index}",
                type=f"FIGHT_MOVE_{index}",
                why=f"Use {move.name} ({move.pp} PP).",
                priority=priority,
                expected_success_signal="Battle text advances or the enemy HP changes",
                objective_id=objective_id,
            )
            self._runtime[candidate.id] = CandidateRuntime(
                action=self._action_for_move_selection(battle, index),
            )
            candidates.append(candidate)

        if battle.kind == "WILD" and pokeballs:
            ball_index, ball_name = pokeballs[0]
            throw_ball = self._build_item_candidate(
                state=state,
                battle=battle,
                objective_id=objective_id,
                item_index=ball_index,
                item_name=ball_name,
                candidate_id="battle_throw_pokeball",
                candidate_type="THROW_POKEBALL",
                why=f"Try a catch with {ball_name}.",
                priority=84 if catch_decision else 52,
            )
            if throw_ball is not None:
                candidates.append(throw_ball)

        if battle.kind == "WILD" and self._should_run(state, battle):
            candidate = CandidateNextStep(
                id="battle_run",
                type="RUN",
                why="This wild encounter looks low-value.",
                priority=58,
                expected_success_signal="The battle ends or the escape text appears",
                objective_id=objective_id,
            )
            self._runtime[candidate.id] = CandidateRuntime(
                action=self._action_for_main_menu_target(battle, MAIN_MENU_POSITIONS["RUN"], "RUN"),
            )
            candidates.append(candidate)

        if not candidates:
            return [self._fallback_candidate(objective_id, "Battle is active and needs a safe input to continue.")]

        deduped = self._dedupe(candidates)
        deduped.sort(key=lambda item: (-item.priority, item.id))
        return deduped[:4]

    def runtime_map(self) -> dict[str, CandidateRuntime]:
        return dict(self._runtime)

    def record_choice(self, candidate: CandidateNextStep) -> None:
        if candidate.type.startswith("FIGHT_MOVE_"):
            self._tracker.submenu = "FIGHT"
            return
        if candidate.type in {"USE_POTION", "THROW_POKEBALL"}:
            self._tracker.submenu = "BAG"
            return
        if candidate.type == "SWITCH_POKEMON":
            self._tracker.submenu = "POKEMON"
            return
        if candidate.type == "RUN":
            self._tracker.submenu = "RUN"

    def _sync_tracker(self, state: StructuredGameState, battle: BattleContext) -> None:
        battle_key = (battle.kind, battle.enemy_species, battle.enemy_level, battle.player_active_species)
        if self._tracker.battle_key != battle_key:
            self._tracker = BattleTracker(battle_key=battle_key, submenu="MAIN")
            return
        self._tracker.submenu = self._infer_submenu(state, battle)

    def _infer_submenu(self, state: StructuredGameState, battle: BattleContext) -> str:
        if battle.player_active_hp is not None and battle.player_active_hp <= 0:
            return "POKEMON"

        position = battle.battle_menu_position
        if position is None:
            return self._tracker.submenu

        if self._tracker.submenu == "FIGHT":
            move_slots = max(1, len([move for move in battle.available_moves if move.pp >= 0]))
            if 1 <= position <= move_slots:
                return "FIGHT"
            if position in MAIN_MENU_COORDS:
                return "MAIN"

        if self._tracker.submenu in {"BAG", "POKEMON"} and position not in MAIN_MENU_COORDS:
            return self._tracker.submenu

        if position in MAIN_MENU_COORDS:
            return "MAIN"
        return self._tracker.submenu

    def _build_item_candidate(
        self,
        *,
        state: StructuredGameState,
        battle: BattleContext,
        objective_id: str | None,
        item_index: int,
        item_name: str,
        candidate_id: str,
        candidate_type: str,
        why: str,
        priority: int,
    ) -> CandidateNextStep | None:
        action = self._action_for_item_selection(battle, item_index, submenu="BAG")
        if action is None:
            return None
        candidate = CandidateNextStep(
            id=candidate_id,
            type=candidate_type,
            why=why,
            priority=priority,
            expected_success_signal=f"{item_name} is used or the bag cursor changes",
            objective_id=objective_id,
        )
        self._runtime[candidate.id] = CandidateRuntime(action=action)
        return candidate

    def _build_switch_candidate(
        self,
        state: StructuredGameState,
        battle: BattleContext,
        objective_id: str | None,
        *,
        priority: int,
    ) -> CandidateNextStep | None:
        switch_target = self._best_switch_target(state, battle)
        if switch_target is None:
            return None
        target_index, target_name = switch_target
        action = self._action_for_party_selection(battle, target_index)
        candidate = CandidateNextStep(
            id="battle_switch_pokemon",
            type="SWITCH_POKEMON",
            why=f"Switch to {target_name}.",
            priority=priority,
            expected_success_signal="The active Pokemon changes or the party cursor advances",
            objective_id=objective_id,
        )
        self._runtime[candidate.id] = CandidateRuntime(action=action)
        return candidate

    def _fallback_candidate(self, objective_id: str | None, why: str) -> CandidateNextStep:
        candidate = CandidateNextStep(
            id="battle_default",
            type="BATTLE_DEFAULT",
            why=why,
            priority=40,
            expected_success_signal="Battle state changes or ends",
            objective_id=objective_id,
        )
        self._runtime[candidate.id] = CandidateRuntime(
            action=ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="advance battle"),
        )
        return candidate

    def _dedupe(self, candidates: list[CandidateNextStep]) -> list[CandidateNextStep]:
        seen: set[str] = set()
        result: list[CandidateNextStep] = []
        for candidate in candidates:
            if candidate.type in seen:
                continue
            seen.add(candidate.type)
            result.append(candidate)
        return result

    def _hp_ratio(self, hp: int | None, max_hp: int | None) -> float | None:
        if hp is None or max_hp is None or max_hp <= 0:
            return None
        return hp / max_hp

    def _indexed_inventory(self, state: StructuredGameState, allowed_names: frozenset[str]) -> list[tuple[int, str]]:
        return [(index, item.name) for index, item in enumerate(state.inventory) if item.name in allowed_names and item.count > 0]

    def _needs_catch_decision(
        self,
        state: StructuredGameState,
        battle: BattleContext,
        pokeballs: list[tuple[int, str]],
    ) -> bool:
        if battle.kind != "WILD" or not pokeballs or not battle.enemy_species:
            return False
        party_species = {member.name for member in state.party}
        if battle.enemy_species not in party_species:
            return True
        enemy_ratio = self._hp_ratio(battle.enemy_hp, battle.enemy_max_hp)
        return enemy_ratio is not None and enemy_ratio <= 0.35

    def _needs_llm_strategy(self, state: StructuredGameState, battle: BattleContext) -> bool:
        if battle.kind == "TRAINER":
            return True
        return any(token in state.map_name for token in IMPORTANT_BATTLE_MAP_TOKENS)

    def _should_run(self, state: StructuredGameState, battle: BattleContext) -> bool:
        if battle.kind != "WILD":
            return False
        hp_ratio = self._hp_ratio(battle.player_active_hp, battle.player_active_max_hp)
        if hp_ratio is None or hp_ratio < 0.9:
            return False
        if battle.enemy_species in COMMON_WILD_SPECIES:
            return True
        if battle.enemy_level is None or battle.player_active_level is None:
            return False
        return battle.player_active_level - battle.enemy_level >= 5

    def _move_priority(
        self,
        state: StructuredGameState,
        battle: BattleContext,
        move: MoveInfo,
        move_index: int,
        strongest_move_index: int | None,
        important_trainer: bool,
        catch_decision: bool,
    ) -> int:
        if strongest_move_index == move_index and battle.kind == "WILD" and not catch_decision:
            hp_ratio = self._hp_ratio(battle.player_active_hp, battle.player_active_max_hp)
            if hp_ratio is not None and hp_ratio >= 0.95:
                return 98
        if strongest_move_index == move_index:
            return 78 if important_trainer else 80
        return max(42, 72 - (move_index * 4))

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

    def _has_type_disadvantage(self, state: StructuredGameState, battle: BattleContext) -> bool:
        current_types = self._species_types(battle.player_active_species)
        enemy_types = self._species_types(battle.enemy_species)
        if current_types is None or enemy_types is None:
            return False

        current_risk = self._incoming_risk(enemy_types, current_types)
        if current_risk < 2.0:
            return False

        for member in state.party:
            if not member.hp or member.hp <= 0:
                continue
            if member.name == battle.player_active_species:
                continue
            member_types = self._species_types(member.name)
            if member_types is None:
                continue
            bench_risk = self._incoming_risk(enemy_types, member_types)
            if bench_risk <= 1.5 or bench_risk + 0.5 < current_risk:
                return True
        return False

    def _best_switch_target(self, state: StructuredGameState, battle: BattleContext) -> tuple[int, str] | None:
        enemy_types = self._species_types(battle.enemy_species)
        candidates: list[tuple[float, int, str]] = []
        for index, member in enumerate(state.party):
            if not member.hp or member.hp <= 0:
                continue
            if member.name == battle.player_active_species:
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

    def _incoming_risk(self, attacker_types: tuple[str, str] | None, defender_types: tuple[str, str] | None) -> float:
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
            multiplier *= TYPE_EFFECTIVENESS.get((move_type, defender_type), 1.0)
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

    def _action_for_move_selection(self, battle: BattleContext, move_index: int) -> ActionDecision:
        if self._tracker.submenu == "POKEMON":
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="return to main battle menu")
        if self._tracker.submenu == "BAG":
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="leave bag and return to battle menu")
        if self._tracker.submenu != "FIGHT":
            return self._action_for_main_menu_target(battle, MAIN_MENU_POSITIONS["FIGHT"], "FIGHT")

        current = max(1, battle.battle_menu_position or 1)
        if current < move_index:
            return ActionDecision(action=ActionType.MOVE_DOWN, repeat=1, reason="move to stronger attack")
        if current > move_index:
            return ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="move to selected attack")
        return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="confirm chosen attack")

    def _action_for_item_selection(self, battle: BattleContext, item_index: int, *, submenu: str) -> ActionDecision | None:
        if self._tracker.submenu == "FIGHT":
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="leave move list and return to main menu")
        if self._tracker.submenu != submenu:
            return self._action_for_main_menu_target(battle, MAIN_MENU_POSITIONS["BAG"], "BAG")

        current = max(0, battle.battle_menu_position or 0)
        if current < item_index:
            return ActionDecision(action=ActionType.MOVE_DOWN, repeat=1, reason="move to the desired bag item")
        if current > item_index:
            return ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="move to the desired bag item")
        return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="use the highlighted item")

    def _action_for_party_selection(self, battle: BattleContext, target_index: int) -> ActionDecision:
        if self._tracker.submenu == "FIGHT":
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="leave move list and return to main menu")
        if self._tracker.submenu == "BAG":
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason="leave bag and return to battle menu")
        if self._tracker.submenu != "POKEMON":
            return self._action_for_main_menu_target(battle, MAIN_MENU_POSITIONS["POKEMON"], "POKEMON")

        current = max(0, battle.battle_menu_position or 0)
        if current < target_index:
            return ActionDecision(action=ActionType.MOVE_DOWN, repeat=1, reason="move to the next healthy Pokemon")
        if current > target_index:
            return ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason="move to the next healthy Pokemon")
        return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason="confirm Pokemon switch")

    def _action_for_main_menu_target(self, battle: BattleContext, target: int, label: str) -> ActionDecision:
        current = battle.battle_menu_position
        if current not in MAIN_MENU_COORDS:
            return ActionDecision(action=ActionType.PRESS_B, repeat=1, reason=f"return to the {label} menu target")

        if current == target:
            return ActionDecision(action=ActionType.PRESS_A, repeat=1, reason=f"open {label.lower()} from the battle menu")

        current_x, current_y = MAIN_MENU_COORDS[current]
        target_x, target_y = MAIN_MENU_COORDS[target]
        if current_x < target_x:
            return ActionDecision(action=ActionType.MOVE_RIGHT, repeat=1, reason=f"move cursor toward {label.lower()}")
        if current_x > target_x:
            return ActionDecision(action=ActionType.MOVE_LEFT, repeat=1, reason=f"move cursor toward {label.lower()}")
        if current_y < target_y:
            return ActionDecision(action=ActionType.MOVE_DOWN, repeat=1, reason=f"move cursor toward {label.lower()}")
        return ActionDecision(action=ActionType.MOVE_UP, repeat=1, reason=f"move cursor toward {label.lower()}")
