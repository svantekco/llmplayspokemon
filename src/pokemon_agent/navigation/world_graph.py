from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from functools import lru_cache
from importlib.resources import files
import re
from typing import Any

GENERIC_MAP_TOKENS = {
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
    "building",
    "co",
    "lobby",
    "floor",
}


@dataclass(frozen=True, slots=True)
class WorldGraphMap:
    id: int
    symbol: str
    name: str
    slug: str
    width: int
    height: int
    file: str | None
    tileset: str | None
    group: str | None
    anchor_name: str | None
    is_unused: bool
    is_copy: bool
    is_multiplayer: bool
    routing_enabled: bool


@dataclass(frozen=True, slots=True)
class WorldGraphEdge:
    source_map_id: int
    source_symbol: str
    source_name: str
    destination_map_id: int | None
    destination_symbol: str | None
    destination_name: str | None
    kind: str
    direction: str | None = None
    x: int | None = None
    y: int | None = None
    warp_index: int | None = None
    destination_warp_id: int | None = None
    original_destination_symbol: str | None = None
    resolution_method: str | None = None


@dataclass(frozen=True, slots=True)
class Landmark:
    id: str
    type: str
    label: str
    map_id: int
    map_symbol: str
    map_name: str
    x: int | None = None
    y: int | None = None
    direction: str | None = None
    destination_map_id: int | None = None
    destination_symbol: str | None = None
    destination_name: str | None = None
    warp_index: int | None = None
    text_id: str | None = None


@dataclass(frozen=True, slots=True)
class RoutePlan:
    maps: tuple[WorldGraphMap, ...]
    edges: tuple[WorldGraphEdge, ...]

    def summary(self) -> list[str]:
        return [node.symbol for node in self.maps]

    def names(self) -> list[str]:
        return [node.name for node in self.maps]


@dataclass(frozen=True, slots=True)
class LandmarkRoute:
    landmark: Landmark
    route: RoutePlan


class WorldGraph:
    def __init__(self, world_graph_payload: dict[str, Any], landmark_payload: dict[str, Any]) -> None:
        self.meta = dict(world_graph_payload.get("meta", {}))
        self.landmark_meta = dict(landmark_payload.get("meta", {}))

        self._maps_by_id: dict[int, WorldGraphMap] = {}
        self._maps_by_symbol: dict[str, WorldGraphMap] = {}
        self._maps_by_slug: dict[str, WorldGraphMap] = {}
        self._maps_by_normalized_name: dict[str, WorldGraphMap] = {}
        self._edges_by_symbol: dict[str, list[WorldGraphEdge]] = {}
        self._warp_lookup: dict[tuple[str, int, int], WorldGraphEdge] = {}
        self._landmarks_by_id: dict[str, Landmark] = {}
        self._landmarks_by_map_symbol: dict[str, list[Landmark]] = {}
        self._landmarks_by_type: dict[str, list[Landmark]] = {}

        for raw_map in world_graph_payload.get("maps", []):
            graph_map = WorldGraphMap(
                id=int(raw_map["id"]),
                symbol=str(raw_map["symbol"]),
                name=str(raw_map["name"]),
                slug=str(raw_map["slug"]),
                width=int(raw_map["width"]),
                height=int(raw_map["height"]),
                file=_as_optional_str(raw_map.get("file")),
                tileset=_as_optional_str(raw_map.get("tileset")),
                group=_as_optional_str(raw_map.get("group")),
                anchor_name=_as_optional_str(raw_map.get("anchor_name")),
                is_unused=bool(raw_map.get("is_unused", False)),
                is_copy=bool(raw_map.get("is_copy", False)),
                is_multiplayer=bool(raw_map.get("is_multiplayer", False)),
                routing_enabled=bool(raw_map.get("routing_enabled", False)),
            )
            self._maps_by_id[graph_map.id] = graph_map
            self._maps_by_symbol[graph_map.symbol] = graph_map
            self._maps_by_slug[graph_map.slug] = graph_map
            self._maps_by_normalized_name[_normalize_name(graph_map.name)] = graph_map
            self._edges_by_symbol[graph_map.symbol] = []

        for raw_map in world_graph_payload.get("maps", []):
            source = self._maps_by_symbol[str(raw_map["symbol"])]
            for raw_connection in raw_map.get("connections", []):
                edge = WorldGraphEdge(
                    source_map_id=source.id,
                    source_symbol=source.symbol,
                    source_name=source.name,
                    destination_map_id=_as_optional_int(raw_connection.get("destination_map_id")),
                    destination_symbol=_as_optional_str(raw_connection.get("destination_symbol")),
                    destination_name=_as_optional_str(raw_connection.get("destination_name")),
                    kind="connection",
                    direction=_as_optional_str(raw_connection.get("direction")),
                )
                self._edges_by_symbol[source.symbol].append(edge)
            for raw_warp in raw_map.get("warps", []):
                edge = WorldGraphEdge(
                    source_map_id=source.id,
                    source_symbol=source.symbol,
                    source_name=source.name,
                    destination_map_id=_as_optional_int(raw_warp.get("destination_map_id")),
                    destination_symbol=_as_optional_str(raw_warp.get("destination_symbol")),
                    destination_name=_as_optional_str(raw_warp.get("destination_name")),
                    kind="warp",
                    x=_as_optional_int(raw_warp.get("x")),
                    y=_as_optional_int(raw_warp.get("y")),
                    warp_index=_as_optional_int(raw_warp.get("index")),
                    destination_warp_id=_as_optional_int(raw_warp.get("destination_warp_id")),
                    original_destination_symbol=_as_optional_str(raw_warp.get("original_destination_symbol")),
                    resolution_method=_as_optional_str(raw_warp.get("resolution_method")),
                )
                self._edges_by_symbol[source.symbol].append(edge)
                if edge.x is not None and edge.y is not None:
                    self._warp_lookup[(source.symbol, edge.x, edge.y)] = edge

        for landmark_payload_item in landmark_payload.get("landmarks", []):
            landmark = Landmark(
                id=str(landmark_payload_item["id"]),
                type=str(landmark_payload_item["type"]),
                label=str(landmark_payload_item["label"]),
                map_id=int(landmark_payload_item["map_id"]),
                map_symbol=str(landmark_payload_item["map_symbol"]),
                map_name=str(landmark_payload_item["map_name"]),
                x=_as_optional_int(landmark_payload_item.get("x")),
                y=_as_optional_int(landmark_payload_item.get("y")),
                direction=_as_optional_str(landmark_payload_item.get("direction")),
                destination_map_id=_as_optional_int(landmark_payload_item.get("destination_map_id")),
                destination_symbol=_as_optional_str(landmark_payload_item.get("destination_symbol")),
                destination_name=_as_optional_str(landmark_payload_item.get("destination_name")),
                warp_index=_as_optional_int(landmark_payload_item.get("warp_index")),
                text_id=_as_optional_str(landmark_payload_item.get("text_id")),
            )
            self._landmarks_by_id[landmark.id] = landmark
            self._landmarks_by_map_symbol.setdefault(landmark.map_symbol, []).append(landmark)
            self._landmarks_by_type.setdefault(landmark.type, []).append(landmark)

        for map_symbol in self._edges_by_symbol:
            self._edges_by_symbol[map_symbol].sort(key=self._edge_sort_key)
        for map_symbol in self._landmarks_by_map_symbol:
            self._landmarks_by_map_symbol[map_symbol].sort(key=lambda item: (item.type, item.id))
        for landmark_type in self._landmarks_by_type:
            self._landmarks_by_type[landmark_type].sort(key=lambda item: (item.map_id, item.id))

    def get_map_by_id(self, map_id: int | str | None) -> WorldGraphMap | None:
        symbol = self._resolve_symbol(map_id)
        if symbol is None:
            return None
        return self._maps_by_symbol.get(symbol)

    getMapById = get_map_by_id

    def get_map_by_name(self, map_name: str | None) -> WorldGraphMap | None:
        symbol = self._resolve_symbol(map_name)
        if symbol is None:
            return None
        return self._maps_by_symbol.get(symbol)

    getMapByName = get_map_by_name

    def neighbors(self, map_ref: int | str | None) -> tuple[WorldGraphEdge, ...]:
        symbol = self._resolve_symbol(map_ref)
        if symbol is None:
            return ()
        return tuple(self._edges_by_symbol.get(symbol, ()))

    def maps(self) -> tuple[WorldGraphMap, ...]:
        return tuple(sorted(self._maps_by_symbol.values(), key=lambda item: item.id))

    def get_warp_at(self, map_ref: int | str | None, x: int, y: int) -> WorldGraphEdge | None:
        symbol = self._resolve_symbol(map_ref)
        if symbol is None:
            return None
        edge = self._warp_lookup.get((symbol, x, y))
        if edge is None or edge.kind != "warp":
            return None
        return edge

    getWarpAt = get_warp_at

    def get_landmark(self, landmark_id: str | None) -> Landmark | None:
        if landmark_id is None:
            return None
        return self._landmarks_by_id.get(landmark_id)

    getLandmark = get_landmark

    def get_landmarks_on_map(self, map_ref: int | str | None) -> tuple[Landmark, ...]:
        symbol = self._resolve_symbol(map_ref)
        if symbol is None:
            return ()
        return tuple(self._landmarks_by_map_symbol.get(symbol, ()))

    getLandmarksOnMap = get_landmarks_on_map

    def nearest_landmark(self, current_map: int | str | None, landmark_type: str) -> LandmarkRoute | None:
        source_symbol = self._resolve_symbol(current_map)
        if source_symbol is None:
            return None
        candidates = self._landmarks_by_type.get(str(landmark_type), [])
        if not candidates:
            return None
        best: tuple[int, str, LandmarkRoute] | None = None
        for landmark in candidates:
            route = self.find_route(source_symbol, landmark.map_symbol)
            if route is None:
                continue
            rank = (len(route.edges), landmark.id)
            payload = LandmarkRoute(landmark=landmark, route=route)
            if best is None or rank < (best[0], best[1]):
                best = (rank[0], rank[1], payload)
        return None if best is None else best[2]

    nearestLandmark = nearest_landmark

    def find_route(self, from_map: int | str | None, to_map: int | str | None) -> RoutePlan | None:
        source_symbol = self._resolve_symbol(from_map)
        target_symbol = self._resolve_symbol(to_map)
        if source_symbol is None or target_symbol is None:
            return None
        source = self._maps_by_symbol[source_symbol]
        target = self._maps_by_symbol[target_symbol]
        if source.symbol == target.symbol:
            return RoutePlan(maps=(source,), edges=())

        queue: deque[str] = deque([source.symbol])
        previous: dict[str, tuple[str, WorldGraphEdge]] = {}
        visited = {source.symbol}

        while queue:
            current_symbol = queue.popleft()
            for edge in self._edges_by_symbol.get(current_symbol, ()):
                destination_symbol = edge.destination_symbol
                if destination_symbol is None:
                    continue
                destination_map = self._maps_by_symbol.get(destination_symbol)
                if destination_map is None or not destination_map.routing_enabled:
                    continue
                if destination_symbol in visited:
                    continue
                visited.add(destination_symbol)
                previous[destination_symbol] = (current_symbol, edge)
                if destination_symbol == target.symbol:
                    queue.clear()
                    break
                queue.append(destination_symbol)

        if target.symbol not in previous:
            return None

        edge_path: list[WorldGraphEdge] = []
        map_path: list[WorldGraphMap] = [target]
        current_symbol = target.symbol
        while current_symbol != source.symbol:
            previous_symbol, edge = previous[current_symbol]
            edge_path.append(edge)
            map_path.append(self._maps_by_symbol[previous_symbol])
            current_symbol = previous_symbol
        edge_path.reverse()
        map_path.reverse()
        return RoutePlan(maps=tuple(map_path), edges=tuple(edge_path))

    findRoute = find_route

    def find_route_to_landmark(self, current_map: int | str | None, landmark_type_or_id: str) -> LandmarkRoute | None:
        landmark = self._landmarks_by_id.get(landmark_type_or_id)
        if landmark is not None:
            route = self.find_route(current_map, landmark.map_symbol)
            if route is None:
                return None
            return LandmarkRoute(landmark=landmark, route=route)
        return self.nearest_landmark(current_map, landmark_type_or_id)

    findRouteToLandmark = find_route_to_landmark

    def canonical_symbol(self, map_ref: int | str | None) -> str | None:
        return self._resolve_symbol(map_ref)

    def canonical_name(self, map_ref: int | str | None) -> str | None:
        graph_map = self.get_map_by_id(map_ref)
        return None if graph_map is None else graph_map.name

    def route_summary(self, route: RoutePlan | None, *, use_names: bool = False) -> list[str] | None:
        if route is None:
            return None
        return route.names() if use_names else route.summary()

    def _resolve_symbol(self, map_ref: int | str | None) -> str | None:
        if map_ref is None:
            return None
        if isinstance(map_ref, int):
            graph_map = self._maps_by_id.get(map_ref)
            return None if graph_map is None else graph_map.symbol
        raw = str(map_ref).strip()
        if not raw:
            return None
        if raw in self._maps_by_symbol:
            return raw
        normalized = _normalize_name(raw)
        if normalized in self._maps_by_slug:
            return self._maps_by_slug[normalized].symbol
        if normalized in self._maps_by_normalized_name:
            return self._maps_by_normalized_name[normalized].symbol
        for graph_map in self._maps_by_symbol.values():
            if _map_matches(graph_map.name, raw) or _map_matches(graph_map.symbol, raw):
                return graph_map.symbol
        return None

    @staticmethod
    def _edge_sort_key(edge: WorldGraphEdge) -> tuple[int, str, str, int, int]:
        return (
            0 if edge.kind == "connection" else 1,
            edge.direction or "",
            edge.destination_symbol or "",
            edge.y if edge.y is not None else -1,
            edge.x if edge.x is not None else -1,
        )


@lru_cache(maxsize=1)
def load_world_graph() -> WorldGraph:
    world_graph_payload = json.loads(files("pokemon_agent.generated").joinpath("world_graph.json").read_text(encoding="utf-8"))
    landmark_payload = json.loads(files("pokemon_agent.generated").joinpath("landmarks.json").read_text(encoding="utf-8"))
    return WorldGraph(world_graph_payload, landmark_payload)


def get_map_by_id(map_id: int | str | None) -> WorldGraphMap | None:
    return load_world_graph().get_map_by_id(map_id)


def get_map_by_name(map_name: str | None) -> WorldGraphMap | None:
    return load_world_graph().get_map_by_name(map_name)


def neighbors(map_ref: int | str | None) -> tuple[WorldGraphEdge, ...]:
    return load_world_graph().neighbors(map_ref)


def get_warp_at(map_ref: int | str | None, x: int, y: int) -> WorldGraphEdge | None:
    return load_world_graph().get_warp_at(map_ref, x, y)


def get_landmark(landmark_id: str | None) -> Landmark | None:
    return load_world_graph().get_landmark(landmark_id)


def get_landmarks_on_map(map_ref: int | str | None) -> tuple[Landmark, ...]:
    return load_world_graph().get_landmarks_on_map(map_ref)


def nearest_landmark(current_map: int | str | None, landmark_type: str) -> LandmarkRoute | None:
    return load_world_graph().nearest_landmark(current_map, landmark_type)


def find_route(from_map: int | str | None, to_map: int | str | None) -> RoutePlan | None:
    return load_world_graph().find_route(from_map, to_map)


def find_route_to_landmark(current_map: int | str | None, landmark_type_or_id: str) -> LandmarkRoute | None:
    return load_world_graph().find_route_to_landmark(current_map, landmark_type_or_id)


def map_matches(current_map_name: str, target_map_name: str) -> bool:
    return _map_matches(current_map_name, target_map_name)


def _normalize_name(value: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", value.lower()))


def _map_matches(current_map_name: str, target_map_name: str) -> bool:
    current_tokens = _tokenize_name(current_map_name)
    target_tokens = _tokenize_name(target_map_name)
    if not current_tokens or not target_tokens:
        return False
    if target_tokens.issubset(current_tokens) or current_tokens.issubset(target_tokens):
        return True

    current_core = current_tokens - GENERIC_MAP_TOKENS
    target_core = target_tokens - GENERIC_MAP_TOKENS
    if not current_core or not target_core:
        return False
    return current_core.issubset(target_core) or target_core.issubset(current_core)


def _tokenize_name(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token}


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
