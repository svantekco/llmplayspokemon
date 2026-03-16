from __future__ import annotations

import numpy as np

import pokemon_agent.agent.menu_manager as menu_manager_module
from pokemon_agent.agent.controllers.menu import MenuController
from pokemon_agent.agent.controllers.protocol import TurnContext
from pokemon_agent.data.walkthrough import Milestone
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import InventoryItem
from pokemon_agent.models.state import PartyMember
from pokemon_agent.models.state import StructuredGameState

_ENCODE = {" ": 0x7F, "'": 0xE0, "!": 0xE7, "?": 0xE6}
_ENCODE.update({letter: 0x80 + index for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
_ENCODE.update({letter: 0xA0 + index for index, letter in enumerate("abcdefghijklmnopqrstuvwxyz")})


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
    labels: list[str] | None = None,
    cursor_index: int = 0,
    top_x: int = 12,
    top_y: int = 2,
    menu_open: bool = True,
    inventory: list[InventoryItem] | None = None,
    story_flags: list[str] | None = None,
) -> StructuredGameState:
    labels = labels or []
    return StructuredGameState(
        map_name="Vermilion City",
        map_id="vermilion_city",
        x=5,
        y=5,
        mode=GameMode.MENU if menu_open else GameMode.OVERWORLD,
        menu_open=menu_open,
        inventory=inventory or [],
        story_flags=story_flags or [],
        party=[PartyMember(name="Charmander", hp=30, max_hp=39)],
        game_area=_menu_grid(labels, top_x=top_x, top_y=top_y) if labels else None,
        metadata={
            "ram_context": {
                "ui": {
                    "window_y": 0,
                    "top_menu_item_x": top_x,
                    "top_menu_item_y": top_y,
                    "current_menu_item": cursor_index,
                    "max_menu_item": len(labels) - 1 if labels else 0,
                }
            }
        },
    )


def test_menu_controller_closes_generic_menu() -> None:
    controller = MenuController()

    result = controller.step(_menu_state(labels=[]), TurnContext(turn_index=1))

    assert result.action is not None
    assert result.action.action == ActionType.PRESS_B
    assert result.planner_source == "menu_controller"


def test_menu_controller_uses_cursor_navigation_for_hm_flow(monkeypatch) -> None:
    monkeypatch.setattr(
        menu_manager_module,
        "get_current_milestone",
        lambda *args, **kwargs: Milestone(
            id="gym3_surge",
            description="Teach Cut and enter the Vermilion Gym.",
            completion_flag=None,
            completion_item=None,
            prerequisite_flags=["got_hm01_cut"],
            prerequisite_items=[],
            target_map_name="Vermilion City",
            route_hints=["Open the menu and prepare Cut."],
            sub_steps=["Teach Cut if needed.", "Use Cut near the gym tree."],
            required_hms=["Cut"],
            next_milestone_id=None,
        ),
    )
    controller = MenuController()
    state = _menu_state(
        labels=["POKEDEX", "POKEMON", "ITEM", "SAVE", "EXIT"],
        cursor_index=0,
        inventory=[InventoryItem(name="HM Cut", count=1)],
        story_flags=["got_hm01_cut"],
    )

    result = controller.step(state, TurnContext(turn_index=1))

    assert result.action is not None
    assert result.action.action == ActionType.MOVE_DOWN


def test_menu_controller_clears_menu_transition() -> None:
    controller = MenuController()

    result = controller.step(_menu_state(menu_open=False), TurnContext(turn_index=1))

    assert result.action is not None
    assert result.action.action == ActionType.PRESS_B
