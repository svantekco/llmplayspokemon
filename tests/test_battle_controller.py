from __future__ import annotations

import json

from pokemon_agent.agent.controllers.battle import BattleController
from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.agent.llm_client import CompletionResponse
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import BattleContext
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import InventoryItem
from pokemon_agent.models.state import MoveInfo
from pokemon_agent.models.state import PartyMember
from pokemon_agent.models.state import StructuredGameState


class _FakeBattleLLM:
    def __init__(self, *payloads: dict) -> None:
        self._payloads = list(payloads)
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], purpose: str) -> CompletionResponse:
        del purpose
        self.calls += 1
        self.messages.append(messages)
        payload = self._payloads[min(self.calls - 1, len(self._payloads) - 1)]
        return CompletionResponse(content=json.dumps(payload), model="fake")


class _FakeKnowledge:
    def __init__(self, overrides: dict[tuple[str, str], float] | None = None) -> None:
        self._overrides = overrides or {}

    def type_effectiveness(self, attack_type: str | None, defend_type: str | None) -> float:
        if attack_type is None or defend_type is None:
            return 1.0
        return self._overrides.get((attack_type, defend_type), 1.0)


def _battle_state(
    *,
    kind: str = "WILD",
    enemy_species: str = "Pidgey",
    enemy_level: int = 6,
    player_species: str = "Pikachu",
    player_level: int = 12,
    player_hp: int = 35,
    player_max_hp: int = 35,
    battle_menu_position: int | None = 0,
    move_cursor_position: int | None = None,
    moves: list[MoveInfo] | None = None,
    party: list[PartyMember] | None = None,
    inventory: list[InventoryItem] | None = None,
) -> StructuredGameState:
    battle = BattleContext(
        kind=kind,
        opponent=enemy_species,
        opponent_level=enemy_level,
        enemy_species=enemy_species,
        enemy_level=enemy_level,
        enemy_hp=20,
        enemy_max_hp=20,
        player_active_species=player_species,
        player_active_level=player_level,
        player_active_hp=player_hp,
        player_active_max_hp=player_max_hp,
        available_moves=moves
        or [
            MoveInfo(move_id=1, name="Quick Attack", pp=30, power=40, move_type="NORMAL"),
            MoveInfo(move_id=2, name="ThunderShock", pp=30, power=40, move_type="ELECTRIC"),
        ],
        battle_menu_position=battle_menu_position,
        move_cursor_position=move_cursor_position,
    )
    return StructuredGameState(
        map_name="Route 1",
        map_id="route_1",
        mode=GameMode.BATTLE,
        battle_state=battle,
        party=party
        or [
            PartyMember(name=player_species, level=player_level, hp=player_hp, max_hp=player_max_hp),
            PartyMember(name="Bulbasaur", level=10, hp=32, max_hp=32),
        ],
        inventory=inventory or [],
    )


def test_battle_controller_skips_llm_for_trivial_wild() -> None:
    llm = _FakeBattleLLM({"should_run": False})
    controller = BattleController(llm.complete)
    state = _battle_state(
        enemy_species="Rattata",
        enemy_level=4,
        player_level=12,
        battle_menu_position=3,
        inventory=[],
        party=[
            PartyMember(name="Pikachu", level=12, hp=35, max_hp=35),
            PartyMember(name="Pidgey", level=11, hp=30, max_hp=30),
        ],
    )

    result = controller.step(state, TurnContext(turn_index=1))

    assert llm.calls == 0
    assert result.llm_attempted is False
    assert result.action is not None
    assert result.action.action == ActionType.PRESS_A


def test_battle_controller_uses_llm_once_per_opponent() -> None:
    llm = _FakeBattleLLM(
        {
            "preferred_moves": ["Quick Attack"],
            "use_items": False,
            "should_catch": True,
            "should_run": False,
            "switch_threshold_hp_pct": 20,
        },
        {
            "preferred_moves": ["ThunderShock"],
            "use_items": False,
            "should_catch": False,
            "should_run": False,
            "switch_threshold_hp_pct": 20,
        },
    )
    controller = BattleController(llm.complete)
    state = _battle_state(
        enemy_species="Abra",
        enemy_level=12,
        battle_menu_position=2,
        inventory=[InventoryItem(name="Poke Ball", count=5)],
    )

    first = controller.step(state, TurnContext(turn_index=1))
    second = controller.step(state, TurnContext(turn_index=2))
    next_state = state.model_copy(deep=True)
    next_state.battle_state = next_state.battle_state.model_copy(update={"enemy_species": "Onix", "enemy_level": 14})
    third = controller.step(next_state, TurnContext(turn_index=3))

    assert llm.calls == 2
    assert first.llm_attempted is True
    assert first.action is not None
    assert first.action.action == ActionType.PRESS_A
    assert second.llm_attempted is False
    assert third.llm_attempted is True


def test_battle_controller_falls_back_to_heuristic_move_scoring() -> None:
    controller = BattleController()
    state = _battle_state(kind="TRAINER", enemy_species="Pidgey", enemy_level=12, battle_menu_position=0)

    first = controller.step(state, TurnContext(turn_index=1))
    follow_up_state = state.model_copy(deep=True)
    follow_up_state.battle_state = follow_up_state.battle_state.model_copy(
        update={"battle_menu_position": 1, "move_cursor_position": 1}
    )
    second = controller.step(follow_up_state, TurnContext(turn_index=2))

    assert first.action is not None
    assert first.action.action == ActionType.PRESS_A
    assert second.action is not None
    assert second.action.action == ActionType.MOVE_DOWN


def test_battle_controller_skips_empty_preferred_moves() -> None:
    llm = _FakeBattleLLM(
        {
            "preferred_moves": ["ThunderShock", "Quick Attack"],
            "use_items": False,
            "switch_threshold_hp_pct": 20,
        }
    )
    controller = BattleController(llm.complete)
    state = _battle_state(
        kind="TRAINER",
        battle_menu_position=0,
        moves=[
            MoveInfo(move_id=1, name="Quick Attack", pp=30, power=40, move_type="NORMAL"),
            MoveInfo(move_id=2, name="ThunderShock", pp=0, power=40, move_type="ELECTRIC"),
        ],
    )

    first = controller.step(state, TurnContext(turn_index=1))
    follow_up_state = state.model_copy(deep=True)
    follow_up_state.battle_state = follow_up_state.battle_state.model_copy(
        update={"battle_menu_position": 1, "move_cursor_position": 1}
    )
    second = controller.step(follow_up_state, TurnContext(turn_index=2))

    assert first.action is not None
    assert first.action.action == ActionType.PRESS_A
    assert second.action is not None
    assert second.action.action == ActionType.PRESS_A


def test_battle_controller_switches_when_strategy_requests_it() -> None:
    llm = _FakeBattleLLM(
        {
            "preferred_moves": ["Quick Attack"],
            "switch_threshold_hp_pct": 50,
            "switch_target": "Bulbasaur",
            "use_items": False,
            "should_run": False,
        }
    )
    controller = BattleController(llm.complete)
    state = _battle_state(
        kind="TRAINER",
        player_hp=10,
        player_max_hp=40,
        battle_menu_position=1,
        inventory=[InventoryItem(name="Potion", count=1)],
    )

    result = controller.step(state, TurnContext(turn_index=1))

    assert result.action is not None
    assert result.action.action == ActionType.PRESS_A


def test_battle_controller_moves_toward_fight_from_bag_cursor() -> None:
    controller = BattleController()
    state = _battle_state(kind="TRAINER", battle_menu_position=2)

    result = controller.step(state, TurnContext(turn_index=1))

    assert result.action is not None
    assert result.action.action == ActionType.MOVE_LEFT


def test_battle_controller_uses_game_knowledge_for_type_chart() -> None:
    knowledge = _FakeKnowledge({("WATER", "GRASS"): 5.0, ("NORMAL", "GRASS"): 1.0})
    controller = BattleController(knowledge=knowledge)
    state = _battle_state(
        kind="TRAINER",
        enemy_species="Bulbasaur",
        player_species="Squirtle",
        battle_menu_position=0,
        moves=[
            MoveInfo(move_id=1, name="Tackle", pp=35, power=80, move_type="NORMAL"),
            MoveInfo(move_id=2, name="Water Gun", pp=25, power=40, move_type="WATER"),
        ],
        party=[
            PartyMember(name="Squirtle", level=12, hp=35, max_hp=35),
            PartyMember(name="Pidgey", level=10, hp=28, max_hp=28),
        ],
    )

    first = controller.step(state, TurnContext(turn_index=1))
    follow_up_state = state.model_copy(deep=True)
    follow_up_state.battle_state = follow_up_state.battle_state.model_copy(
        update={"battle_menu_position": 1, "move_cursor_position": 1}
    )
    second = controller.step(follow_up_state, TurnContext(turn_index=2))

    assert first.action is not None
    assert first.action.action == ActionType.PRESS_A
    assert second.action is not None
    assert second.action.action == ActionType.MOVE_DOWN
