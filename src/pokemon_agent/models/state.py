from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class GameMode(str, Enum):
    OVERWORLD = "OVERWORLD"
    MENU = "MENU"
    TEXT = "TEXT"
    BATTLE = "BATTLE"
    CUTSCENE = "CUTSCENE"
    UNKNOWN = "UNKNOWN"


class PartyMember(BaseModel):
    name: str
    species_id: int | None = None
    level: int | None = None
    hp: int | None = None
    max_hp: int | None = None
    status: str | None = None


class InventoryItem(BaseModel):
    name: str
    item_id: int | None = None
    count: int = 1


class MoveInfo(BaseModel):
    move_id: int
    name: str
    pp: int
    power: int | None = None
    move_type: str | None = None


class BattleContext(BaseModel):
    kind: str
    opponent: str | None = None
    opponent_level: int | None = None
    moves: list[str] = Field(default_factory=list)
    enemy_species: str | None = None
    enemy_level: int | None = None
    enemy_hp: int | None = None
    enemy_max_hp: int | None = None
    player_active_species: str | None = None
    player_active_level: int | None = None
    player_active_hp: int | None = None
    player_active_max_hp: int | None = None
    available_moves: list[MoveInfo] = Field(default_factory=list)
    battle_menu_state: int | None = None
    battle_menu_position: int | None = None
    move_cursor_position: int | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class WorldCoordinate(BaseModel):
    x: int
    y: int


class NavigationSnapshot(BaseModel):
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    player: WorldCoordinate
    walkable: list[WorldCoordinate] = Field(default_factory=list)
    blocked: list[WorldCoordinate] = Field(default_factory=list)
    collision_hash: str | None = None


class StructuredGameState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    map_name: str = "UNKNOWN"
    map_id: str | int | None = None
    x: int | None = None
    y: int | None = None
    facing: str | None = None
    mode: GameMode = GameMode.UNKNOWN
    menu_open: bool = False
    text_box_open: bool = False
    battle_state: BattleContext | None = None
    navigation: NavigationSnapshot | None = None
    party: list[PartyMember] = Field(default_factory=list)
    inventory: list[InventoryItem] = Field(default_factory=list)
    story_flags: list[str] = Field(default_factory=list)
    badges: list[str] = Field(default_factory=list)
    step: int = 0
    game_area: list[list[int]] | None = Field(default=None, exclude=True, repr=False)
    collision_area: list[list[int]] | None = Field(default=None, exclude=True, repr=False)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_bootstrap(self) -> bool:
        return self.metadata.get("engine_phase") == "bootstrap"

    def bootstrap_phase(self) -> str | None:
        value = self.metadata.get("bootstrap_phase")
        return value if isinstance(value, str) and value else None

    def state_signature(self) -> tuple:
        battle_kind = None
        if self.battle_state:
            battle_kind = self.battle_state.kind
        return (
            self.map_id,
            self.map_name,
            self.x,
            self.y,
            self.facing,
            self.mode.value,
            self.menu_open,
            self.text_box_open,
            battle_kind,
        )

    def prompt_summary(self) -> dict[str, Any]:
        return {
            "map_name": self.map_name,
            "map_id": self.map_id,
            "position": {"x": self.x, "y": self.y},
            "facing": self.facing,
            "mode": self.mode.value,
            "menu_open": self.menu_open,
            "text_box_open": self.text_box_open,
            "battle_state": self.battle_state.model_dump() if self.battle_state is not None else None,
            "navigation": self.navigation.model_dump() if self.navigation is not None else None,
            "party": [member.model_dump() for member in self.party],
            "inventory": [item.model_dump() for item in self.inventory],
            "story_flags": list(self.story_flags),
            "badges": list(self.badges),
            "step": self.step,
            "metadata": self.metadata,
        }
