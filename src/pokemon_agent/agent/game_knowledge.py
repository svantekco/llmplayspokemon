from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
from typing import Any

from pokemon_agent.navigation.world_graph import WorldGraph
from pokemon_agent.navigation.world_graph import load_world_graph


@dataclass(frozen=True, slots=True)
class WarpDef:
    x: int
    y: int
    dest_map: str | None
    dest_warp_index: int | None
    activation_mode: str | None = None
    destination_map_id: int | None = None
    destination_symbol: str | None = None
    original_destination_symbol: str | None = None
    resolution_method: str | None = None
    tile_id: int | None = None


@dataclass(frozen=True, slots=True)
class NPCDef:
    sprite_id: int | None
    x: int
    y: int
    is_trainer: bool
    text_id: str | None = None
    sprite_symbol: str | None = None
    trainer_class: str | None = None
    trainer_id: int | None = None


@dataclass(frozen=True, slots=True)
class SignDef:
    x: int
    y: int
    text_id: str | None = None


class GameKnowledge:
    def __init__(
        self,
        *,
        world_graph: WorldGraph | None = None,
        map_objects_payload: dict[str, Any] | None = None,
        type_chart_payload: dict[str, Any] | None = None,
    ) -> None:
        self._world_graph = world_graph or load_world_graph()
        self._map_objects_payload = map_objects_payload or _load_generated_json("map_objects.json")
        self._type_chart_payload = type_chart_payload or _load_generated_json("type_chart.json")

        self.meta = dict(self._map_objects_payload.get("meta", {}))
        self.type_chart_meta = dict(self._type_chart_payload.get("meta", {}))

        self._warps_by_map: dict[str, tuple[WarpDef, ...]] = {}
        self._warps_by_coordinate: dict[tuple[str, int, int], WarpDef] = {}
        self._npcs_by_map: dict[str, tuple[NPCDef, ...]] = {}
        self._signs_by_map: dict[str, tuple[SignDef, ...]] = {}
        self._effectiveness: dict[tuple[str, str], float] = {}

        for map_payload in self._map_objects_payload.get("maps", []):
            symbol = str(map_payload["symbol"])
            warps: list[WarpDef] = []
            for raw_warp in map_payload.get("warps", []):
                warp = WarpDef(
                    x=int(raw_warp["x"]),
                    y=int(raw_warp["y"]),
                    dest_map=_optional_str(raw_warp.get("destination_name")),
                    dest_warp_index=_optional_int(raw_warp.get("destination_warp_id")),
                    activation_mode=_optional_str(raw_warp.get("activation_mode")),
                    destination_map_id=_optional_int(raw_warp.get("destination_map_id")),
                    destination_symbol=_optional_str(raw_warp.get("destination_symbol")),
                    original_destination_symbol=_optional_str(raw_warp.get("original_destination_symbol")),
                    resolution_method=_optional_str(raw_warp.get("resolution_method")),
                    tile_id=_optional_int(raw_warp.get("tile_id")),
                )
                warps.append(warp)
                self._warps_by_coordinate[(symbol, warp.x, warp.y)] = warp
            self._warps_by_map[symbol] = tuple(warps)

            self._npcs_by_map[symbol] = tuple(
                NPCDef(
                    sprite_id=_optional_int(raw_npc.get("sprite_id")),
                    x=int(raw_npc["x"]),
                    y=int(raw_npc["y"]),
                    is_trainer=bool(raw_npc.get("is_trainer", False)),
                    text_id=_optional_str(raw_npc.get("text_id")),
                    sprite_symbol=_optional_str(raw_npc.get("sprite_symbol")),
                    trainer_class=_optional_str(raw_npc.get("trainer_class")),
                    trainer_id=_optional_int(raw_npc.get("trainer_id")),
                )
                for raw_npc in map_payload.get("npcs", [])
            )

            self._signs_by_map[symbol] = tuple(
                SignDef(
                    x=int(raw_sign["x"]),
                    y=int(raw_sign["y"]),
                    text_id=_optional_str(raw_sign.get("text_id")),
                )
                for raw_sign in map_payload.get("signs", [])
            )

        for raw_matchup in self._type_chart_payload.get("matchups", []):
            attacker = _normalize_type_name(raw_matchup.get("attacker"))
            defender = _normalize_type_name(raw_matchup.get("defender"))
            if attacker is None or defender is None:
                continue
            self._effectiveness[(attacker, defender)] = float(raw_matchup.get("multiplier", 1.0))

    def get_warp_at(self, map_ref: int | str | None, x: int, y: int) -> WarpDef | None:
        symbol = self._resolve_map_symbol(map_ref)
        if symbol is None:
            return None
        return self._warps_by_coordinate.get((symbol, x, y))

    def get_warps_on_map(self, map_ref: int | str | None) -> list[WarpDef]:
        symbol = self._resolve_map_symbol(map_ref)
        if symbol is None:
            return []
        return list(self._warps_by_map.get(symbol, ()))

    def get_npcs_on_map(self, map_ref: int | str | None) -> list[NPCDef]:
        symbol = self._resolve_map_symbol(map_ref)
        if symbol is None:
            return []
        return list(self._npcs_by_map.get(symbol, ()))

    def get_signs_on_map(self, map_ref: int | str | None) -> list[SignDef]:
        symbol = self._resolve_map_symbol(map_ref)
        if symbol is None:
            return []
        return list(self._signs_by_map.get(symbol, ()))

    def type_effectiveness(self, attack_type: str | None, defend_type: str | None) -> float:
        attacker = _normalize_type_name(attack_type)
        defender = _normalize_type_name(defend_type)
        if attacker is None or defender is None:
            return 1.0
        return self._effectiveness.get((attacker, defender), 1.0)

    def is_super_effective(self, attack_type: str | None, defend_type: str | None) -> bool:
        return self.type_effectiveness(attack_type, defend_type) > 1.0

    def _resolve_map_symbol(self, map_ref: int | str | None) -> str | None:
        graph_map = self._world_graph.get_map_by_id(map_ref)
        if graph_map is None:
            return None
        return graph_map.symbol


@lru_cache(maxsize=1)
def load_game_knowledge() -> GameKnowledge:
    return GameKnowledge()


def _load_generated_json(name: str) -> dict[str, Any]:
    return json.loads(files("pokemon_agent.generated").joinpath(name).read_text(encoding="utf-8"))


def _normalize_type_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper().replace(" ", "_")
    if not normalized:
        return None
    if normalized == "PSYCHIC_TYPE":
        return "PSYCHIC"
    return normalized


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
