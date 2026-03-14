from __future__ import annotations

import numpy as np

from pokemon_agent.agent.menu_manager import MenuManager
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import InventoryItem
from pokemon_agent.models.state import PartyMember
from pokemon_agent.models.state import StructuredGameState


_ENCODE = {" ": 0x7F, "'": 0xE0, "!": 0xE7, "?": 0xE6}
_ENCODE.update({letter: 0x80 + index for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
_ENCODE.update({letter: 0xA0 + index for index, letter in enumerate("abcdefghijklmnopqrstuvwxyz")})
_CUT_PROGRESS_FLAGS = [
    "got_starter",
    "oak_received_parcel",
    "got_pokedex",
    "beat_brock",
    "beat_misty",
    "got_ss_ticket",
    "got_hm01_cut",
]


def _menu_grid(labels: list[str], *, top_x: int, top_y: int) -> list[list[int]]:
    grid = np.full((18, 20), 383, dtype=np.uint32)
    for index, label in enumerate(labels):
        row = top_y + (index * 2)
        if row >= 18:
            break
        for offset, char in enumerate(label[: max(0, 20 - (top_x + 1))], start=top_x + 1):
            grid[row, offset] = _ENCODE.get(char, 0x7F)
    return grid.tolist()


def _menu_state(
    *,
    map_name: str = "Vermilion City",
    menu_open: bool = True,
    mode: GameMode = GameMode.MENU,
    labels: list[str] | None = None,
    cursor_index: int = 0,
    top_x: int = 12,
    top_y: int = 2,
    max_item: int | None = None,
    story_flags: list[str] | None = None,
    inventory: list[InventoryItem] | None = None,
    party: list[PartyMember] | None = None,
) -> StructuredGameState:
    labels = labels or []
    inventory = inventory or []
    party = party or [PartyMember(name="Charmander", hp=30, max_hp=39)]
    story_flags = story_flags or []
    max_item = len(labels) - 1 if max_item is None else max_item
    return StructuredGameState(
        map_name=map_name,
        map_id=map_name.lower().replace(" ", "_"),
        x=5,
        y=5,
        mode=mode,
        menu_open=menu_open,
        story_flags=story_flags,
        inventory=inventory,
        party=party,
        game_area=_menu_grid(labels, top_x=top_x, top_y=top_y) if labels else None,
        metadata={
            "engine_phase": "active",
            "ram_context": {
                "ui": {
                    "window_y": 0,
                    "top_menu_item_x": top_x,
                    "top_menu_item_y": top_y,
                    "current_menu_item": cursor_index,
                    "max_menu_item": max_item,
                }
            },
        },
    )


def test_menu_manager_opens_start_menu_for_required_hm() -> None:
    manager = MenuManager()
    state = _menu_state(
        menu_open=False,
        mode=GameMode.OVERWORLD,
        story_flags=_CUT_PROGRESS_FLAGS,
        inventory=[InventoryItem(name="HM Cut", count=1)],
    )

    candidates = manager.build_candidates(state, "short_gym3_surge")

    assert len(candidates) == 1
    assert candidates[0].type == "OPEN_START_MENU_FOR_HM"
    assert candidates[0].action.action == ActionType.PRESS_START


def test_menu_manager_prioritizes_item_in_start_menu_for_hm_teaching() -> None:
    manager = MenuManager()
    state = _menu_state(
        labels=["POKEDEX", "POKEMON", "ITEM", "SAVE", "EXIT"],
        cursor_index=0,
        story_flags=_CUT_PROGRESS_FLAGS,
        inventory=[InventoryItem(name="HM Cut", count=1)],
    )

    candidates = manager.build_candidates(state, "short_gym3_surge")
    best = max(candidates, key=lambda candidate: (candidate.priority, candidate.id))

    assert best.target is not None
    assert best.target.detail == "ITEM"
    assert best.type == "SELECT_START_MENU_OPTION_FOR_HM"
    assert best.action.action == ActionType.MOVE_DOWN


def test_menu_manager_selects_hm_item_from_item_menu() -> None:
    manager = MenuManager()
    state = _menu_state(
        labels=["Potion", "HM Cut", "Escape Rope"],
        top_x=1,
        top_y=2,
        cursor_index=0,
        story_flags=_CUT_PROGRESS_FLAGS,
        inventory=[
            InventoryItem(name="Potion", count=3),
            InventoryItem(name="HM Cut", count=1),
            InventoryItem(name="Escape Rope", count=1),
        ],
    )

    candidates = manager.build_candidates(state, "short_gym3_surge")
    best = max(candidates, key=lambda candidate: (candidate.priority, candidate.id))

    assert best.type == "SELECT_HM_ITEM"
    assert best.target is not None
    assert best.target.detail == "HM Cut"
    assert best.action.action == ActionType.MOVE_DOWN


def test_menu_manager_uses_field_move_when_visible_in_pokemon_submenu() -> None:
    manager = MenuManager()
    state = _menu_state(
        labels=["Cut", "Stats", "Cancel"],
        top_x=12,
        top_y=8,
        cursor_index=2,
        story_flags=_CUT_PROGRESS_FLAGS,
        inventory=[InventoryItem(name="HM Cut", count=1)],
    )

    candidates = manager.build_candidates(state, "short_gym3_surge")
    best = max(candidates, key=lambda candidate: (candidate.priority, candidate.id))

    assert best.type == "USE_FIELD_MOVE"
    assert best.target is not None
    assert best.target.detail == "Cut"
    assert best.action.action == ActionType.MOVE_UP
