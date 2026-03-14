from __future__ import annotations

from dataclasses import dataclass

from pokemon_agent.navigation.world_graph import find_route
from pokemon_agent.navigation.world_graph import load_world_graph
from pokemon_agent.navigation.world_graph import map_matches as canonical_map_matches


@dataclass(frozen=True, slots=True)
class MapConnection:
    from_map: str
    direction: str
    to_map: str


_WORLD_GRAPH = load_world_graph()


def _build_connection_tables() -> tuple[tuple[MapConnection, ...], tuple[MapConnection, ...]]:
    boundary_connections: list[MapConnection] = []
    warp_connections: list[MapConnection] = []
    for graph_map in _WORLD_GRAPH.maps():
        for edge in _WORLD_GRAPH.neighbors(graph_map.symbol):
            if edge.destination_name is None:
                continue
            connection = MapConnection(from_map=graph_map.name, direction=edge.direction or edge.kind, to_map=edge.destination_name)
            if edge.kind == "connection":
                boundary_connections.append(connection)
            elif edge.kind == "warp":
                warp_connections.append(connection)
    return tuple(boundary_connections), tuple(warp_connections)


MAP_CONNECTIONS, WARP_CONNECTIONS = _build_connection_tables()


def exits_from(map_name: str) -> list[MapConnection]:
    graph_map = _WORLD_GRAPH.get_map_by_name(map_name)
    if graph_map is None:
        return []
    results: list[MapConnection] = []
    for edge in _WORLD_GRAPH.neighbors(graph_map.symbol):
        if edge.destination_name is None:
            continue
        results.append(MapConnection(from_map=graph_map.name, direction=edge.direction or edge.kind, to_map=edge.destination_name))
    return results


def destination_for_exit(map_name: str, direction: str) -> str | None:
    direction_lower = direction.lower()
    for connection in exits_from(map_name):
        if connection.direction.lower() == direction_lower:
            return connection.to_map
    return None


def destinations_for_exit(map_name: str, direction: str) -> list[str]:
    direction_lower = direction.lower()
    return [connection.to_map for connection in exits_from(map_name) if connection.direction.lower() == direction_lower]


def shortest_map_path(from_map: str, to_map: str) -> list[str] | None:
    route = find_route(from_map, to_map)
    if route is None:
        return None
    names = route.names()
    return names[1:] if len(names) > 1 else []


def direction_toward(from_map: str, to_map: str) -> str | None:
    next_hop = next_hop_toward(from_map, to_map)
    return None if next_hop is None else next_hop.direction


def next_hop_toward(from_map: str, to_map: str) -> MapConnection | None:
    route = find_route(from_map, to_map)
    if route is None or not route.edges:
        return None
    first_edge = route.edges[0]
    if first_edge.destination_name is None:
        return None
    return MapConnection(
        from_map=route.maps[0].name,
        direction=first_edge.direction or first_edge.kind,
        to_map=first_edge.destination_name,
    )


def map_matches(current_map_name: str, target_map_name: str) -> bool:
    return canonical_map_matches(current_map_name, target_map_name)
