from __future__ import annotations

from collections import deque
from typing import Iterable

from pokemon_agent.models.action import ActionDecision
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.memory import ConnectorStatus
from pokemon_agent.models.memory import DiscoveredConnector
from pokemon_agent.models.memory import DiscoveredMap
from pokemon_agent.models.memory import NavigationGoal
from pokemon_agent.models.memory import WorldMapMemory
from pokemon_agent.models.state import GameMode
from pokemon_agent.models.state import NavigationSnapshot
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.models.state import WorldCoordinate
from pokemon_agent.agent.navigation import is_real_map_edge
from pokemon_agent.agent.navigation import visible_boundary_side

MOVE_TO_SIDE: dict[ActionType, str] = {
    ActionType.MOVE_UP: "north",
    ActionType.MOVE_RIGHT: "east",
    ActionType.MOVE_DOWN: "south",
    ActionType.MOVE_LEFT: "west",
}
MOVE_DELTAS: dict[ActionType, tuple[int, int]] = {
    ActionType.MOVE_UP: (0, -1),
    ActionType.MOVE_RIGHT: (1, 0),
    ActionType.MOVE_DOWN: (0, 1),
    ActionType.MOVE_LEFT: (-1, 0),
}
OPPOSITE_ACTION: dict[ActionType, ActionType] = {
    ActionType.MOVE_UP: ActionType.MOVE_DOWN,
    ActionType.MOVE_RIGHT: ActionType.MOVE_LEFT,
    ActionType.MOVE_DOWN: ActionType.MOVE_UP,
    ActionType.MOVE_LEFT: ActionType.MOVE_RIGHT,
}
OPPOSITE_SIDE = {
    "north": "south",
    "east": "west",
    "south": "north",
    "west": "east",
}


def observe_state(world_map: WorldMapMemory, state: StructuredGameState) -> None:
    if state.mode != GameMode.OVERWORLD or state.navigation is None or not state.map_name:
        return

    discovered_map = ensure_discovered_map(world_map, state.map_name, state.map_id, state.step)
    discovered_map.walkable = _merge_coordinates(discovered_map.walkable, state.navigation.walkable)
    discovered_map.blocked = _merge_coordinates(discovered_map.blocked, state.navigation.blocked)
    discovered_map.last_seen_step = state.step
    _detect_connectors(world_map, discovered_map, state.navigation, state.step)


def ensure_discovered_map(
    world_map: WorldMapMemory,
    map_name: str,
    map_id: str | int | None,
    step: int | None = None,
) -> DiscoveredMap:
    discovered_map = world_map.maps.get(map_name)
    if discovered_map is None:
        discovered_map = DiscoveredMap(map_name=map_name, map_id=map_id, last_seen_step=step)
        world_map.maps[map_name] = discovered_map
    else:
        if discovered_map.map_id is None and map_id is not None:
            discovered_map.map_id = map_id
        if step is not None:
            discovered_map.last_seen_step = step
    return discovered_map


def confirm_transition(
    world_map: WorldMapMemory,
    previous: StructuredGameState,
    action: ActionDecision,
    current: StructuredGameState,
) -> DiscoveredConnector | None:
    if previous.map_name == current.map_name:
        return None
    if not previous.map_name or not current.map_name:
        return None

    previous_map = ensure_discovered_map(world_map, previous.map_name, previous.map_id, previous.step)
    current_map = ensure_discovered_map(world_map, current.map_name, current.map_id, current.step)

    side = MOVE_TO_SIDE.get(action.action)
    delta = MOVE_DELTAS.get(action.action)
    source_x = None if previous.x is None or delta is None else previous.x + delta[0]
    source_y = None if previous.y is None or delta is None else previous.y + delta[1]

    connector = _match_connector(
        world_map,
        source_map=previous.map_name,
        side=side,
        source_x=source_x,
        source_y=source_y,
    )
    if connector is None:
        connector = _create_connector(
            source_map=previous.map_name,
            side=side,
            source_x=source_x,
            source_y=source_y,
            kind=_kind_for_transition(side, source_x, source_y),
            step=current.step,
        )
        _register_connector(world_map, previous_map, connector)

    connector.status = ConnectorStatus.CONFIRMED
    connector.destination_map = current.map_name
    connector.destination_x = current.x
    connector.destination_y = current.y
    connector.approach_x = previous.x
    connector.approach_y = previous.y
    connector.transition_action = action.action
    connector.confirmed_step = current.step
    if connector.discovered_step is None:
        connector.discovered_step = current.step
    if side and connector.source_side is None:
        connector.source_side = side
    if source_x is not None and connector.source_x is None:
        connector.source_x = source_x
    if source_y is not None and connector.source_y is None:
        connector.source_y = source_y

    reverse_connector = _build_reverse_connector(connector, current)
    if reverse_connector is not None:
        _register_connector(world_map, current_map, reverse_connector)

    return connector


def shortest_confirmed_path(world_map: WorldMapMemory, from_map: str, to_map: str) -> list[DiscoveredConnector] | None:
    if not from_map or not to_map or from_map == to_map:
        return []

    visited: set[str] = {from_map}
    queue: deque[tuple[str, list[str]]] = deque([(from_map, [])])
    while queue:
        current_map, path = queue.popleft()
        if current_map == to_map:
            return [world_map.connectors[connector_id] for connector_id in path if connector_id in world_map.connectors]
        for connector in connectors_from_map(world_map, current_map, confirmed_only=True):
            if not connector.destination_map or connector.destination_map in visited:
                continue
            visited.add(connector.destination_map)
            queue.append((connector.destination_map, [*path, connector.id]))
    return None


def connectors_from_map(
    world_map: WorldMapMemory,
    map_name: str,
    *,
    confirmed_only: bool = False,
) -> list[DiscoveredConnector]:
    discovered_map = world_map.maps.get(map_name)
    if discovered_map is None:
        return []
    connectors: list[DiscoveredConnector] = []
    for connector_id in discovered_map.connectors:
        connector = world_map.connectors.get(connector_id)
        if connector is None:
            continue
        if confirmed_only and connector.status != ConnectorStatus.CONFIRMED:
            continue
        connectors.append(connector)
    connectors.sort(key=lambda item: (item.source_side or "", item.source_y or -1, item.source_x or -1, item.id))
    return connectors


def summarize_navigation_goal(
    world_map: WorldMapMemory,
    current_map: str,
    goal: NavigationGoal | None,
) -> dict[str, object]:
    if goal is None:
        return {}
    summary: dict[str, object] = {
        "target_map": goal.target_map_name,
        "kind": goal.objective_kind,
        "engine_mode": goal.engine_mode,
    }
    if goal.next_map_name is not None:
        summary["next_map"] = goal.next_map_name
    if goal.next_hop_kind is not None:
        summary["next_hop_kind"] = goal.next_hop_kind
    if goal.next_hop_side is not None:
        summary["next_hop_side"] = goal.next_hop_side
    if goal.target_connector_id is not None:
        summary["target_connector_id"] = goal.target_connector_id
    route = shortest_confirmed_path(world_map, current_map, goal.target_map_name)
    if route is None:
        summary["discovered_route_available"] = False
        return summary
    summary["discovered_route_available"] = True
    summary["remaining_known_hops"] = len(route)
    if route:
        next_connector = route[0]
        summary["next_map"] = next_connector.destination_map
        summary["next_connector"] = describe_connector(next_connector)
    else:
        summary["next_map"] = None
        summary["next_connector"] = "already on target map"
    known_destinations = sorted(
        {
            connector.destination_map
            for connector in connectors_from_map(world_map, current_map, confirmed_only=True)
            if connector.destination_map
        }
    )
    if known_destinations:
        summary["known_destinations_on_current_map"] = known_destinations[:6]
    return summary


def world_map_stats(world_map: WorldMapMemory) -> dict[str, int]:
    confirmed = sum(1 for connector in world_map.connectors.values() if connector.status == ConnectorStatus.CONFIRMED)
    suspected = sum(1 for connector in world_map.connectors.values() if connector.status == ConnectorStatus.SUSPECTED)
    return {
        "discovered_maps": len(world_map.maps),
        "confirmed_connectors": confirmed,
        "suspected_connectors": suspected,
    }


def describe_connector(connector: DiscoveredConnector) -> str:
    if connector.destination_map:
        destination = connector.destination_map
    else:
        destination = "unknown"
    if connector.source_side:
        return f"{connector.kind}:{connector.source_side}->{destination}"
    if connector.source_x is not None and connector.source_y is not None:
        return f"{connector.kind}:({connector.source_x}, {connector.source_y})->{destination}"
    return f"{connector.kind}->{destination}"


def render_world_map_preview(world_map: WorldMapMemory) -> str:
    stats = world_map_stats(world_map)
    if stats["discovered_maps"] == 0:
        return "No world map discovered yet."
    lines = [
        f"Maps: {stats['discovered_maps']}",
        f"Confirmed connectors: {stats['confirmed_connectors']}",
        f"Suspected connectors: {stats['suspected_connectors']}",
    ]
    for map_name in sorted(world_map.maps.keys())[:6]:
        destinations = sorted(
            {
                connector.destination_map
                for connector in connectors_from_map(world_map, map_name, confirmed_only=True)
                if connector.destination_map
            }
        )
        destination_text = ", ".join(destinations[:3]) if destinations else "none"
        lines.append(f"{map_name}: {destination_text}")
    return "\n".join(lines)


def _detect_connectors(
    world_map: WorldMapMemory,
    discovered_map: DiscoveredMap,
    navigation: NavigationSnapshot,
    step: int,
) -> None:
    walkable = {(coord.x, coord.y) for coord in navigation.walkable}
    blocked = {(coord.x, coord.y) for coord in navigation.blocked}

    for side, coordinate in _boundary_connectors(navigation):
        if not is_real_map_edge(navigation, side):
            continue
        connector = _create_connector(
            source_map=discovered_map.map_name,
            side=side,
            source_x=coordinate.x,
            source_y=coordinate.y,
            kind="boundary",
            step=step,
        )
        _register_connector(world_map, discovered_map, connector)

    for blocked_x, blocked_y in blocked:
        side = _boundary_side(navigation, blocked_x, blocked_y)
        walkable_neighbors = _adjacent_walkable_count(blocked_x, blocked_y, walkable)
        if side is not None and is_real_map_edge(navigation, side) and walkable_neighbors > 0:
            connector = _create_connector(
                source_map=discovered_map.map_name,
                side=side,
                source_x=blocked_x,
                source_y=blocked_y,
                kind="door",
                step=step,
            )
            _register_connector(world_map, discovered_map, connector)
        elif walkable_neighbors > 0 and _blocked_neighbor_count(blocked_x, blocked_y, blocked) <= 1:
            connector = _create_connector(
                source_map=discovered_map.map_name,
                side=None,
                source_x=blocked_x,
                source_y=blocked_y,
                kind="warp",
                step=step,
            )
            _register_connector(world_map, discovered_map, connector)


def _boundary_connectors(navigation: NavigationSnapshot) -> Iterable[tuple[str, WorldCoordinate]]:
    side_targets: dict[str, WorldCoordinate] = {}
    for coordinate in navigation.walkable:
        side = _boundary_side(navigation, coordinate.x, coordinate.y)
        if side is None:
            continue
        current = side_targets.get(side)
        if current is None or (coordinate.y, coordinate.x) < (current.y, current.x):
            side_targets[side] = coordinate
    return side_targets.items()


def _create_connector(
    *,
    source_map: str,
    side: str | None,
    source_x: int | None,
    source_y: int | None,
    kind: str,
    step: int | None,
) -> DiscoveredConnector:
    return DiscoveredConnector(
        id=_connector_id(source_map, side, source_x, source_y),
        source_map=source_map,
        source_side=side,
        source_x=source_x,
        source_y=source_y,
        kind=kind,
        discovered_step=step,
    )


def _register_connector(world_map: WorldMapMemory, discovered_map: DiscoveredMap, connector: DiscoveredConnector) -> None:
    existing = world_map.connectors.get(connector.id)
    if existing is None:
        world_map.connectors[connector.id] = connector
        stored = connector
    else:
        stored = existing
        if _kind_rank(connector.kind) > _kind_rank(existing.kind):
            existing.kind = connector.kind
        if existing.source_side is None:
            existing.source_side = connector.source_side
        if existing.source_x is None:
            existing.source_x = connector.source_x
        if existing.source_y is None:
            existing.source_y = connector.source_y
        if existing.discovered_step is None:
            existing.discovered_step = connector.discovered_step
        if connector.status == ConnectorStatus.CONFIRMED:
            existing.status = ConnectorStatus.CONFIRMED
        if existing.approach_x is None:
            existing.approach_x = connector.approach_x
        if existing.approach_y is None:
            existing.approach_y = connector.approach_y
        if existing.transition_action is None:
            existing.transition_action = connector.transition_action
        if existing.destination_map is None:
            existing.destination_map = connector.destination_map
        if existing.destination_x is None:
            existing.destination_x = connector.destination_x
        if existing.destination_y is None:
            existing.destination_y = connector.destination_y
        if existing.confirmed_step is None:
            existing.confirmed_step = connector.confirmed_step
    if stored.id not in discovered_map.connectors:
        discovered_map.connectors.append(stored.id)


def _match_connector(
    world_map: WorldMapMemory,
    *,
    source_map: str,
    side: str | None,
    source_x: int | None,
    source_y: int | None,
) -> DiscoveredConnector | None:
    candidates = connectors_from_map(world_map, source_map, confirmed_only=False)
    for connector in candidates:
        if source_x is not None and source_y is not None and connector.source_x == source_x and connector.source_y == source_y:
            return connector
    for connector in candidates:
        if side is not None and connector.source_side == side:
            return connector
    return None


def _build_reverse_connector(connector: DiscoveredConnector, current: StructuredGameState) -> DiscoveredConnector | None:
    if connector.destination_map is None or connector.approach_x is None or connector.approach_y is None:
        return None
    reverse_action = OPPOSITE_ACTION.get(connector.transition_action) if connector.transition_action is not None else None
    reverse_side = OPPOSITE_SIDE.get(connector.source_side or "") or None
    reverse = DiscoveredConnector(
        id=_connector_id(current.map_name, reverse_side, current.x, current.y),
        source_map=current.map_name,
        source_side=reverse_side,
        source_x=current.x,
        source_y=current.y,
        kind=connector.kind,
        status=ConnectorStatus.CONFIRMED,
        approach_x=current.x,
        approach_y=current.y,
        transition_action=reverse_action,
        destination_map=connector.source_map,
        destination_x=connector.approach_x,
        destination_y=connector.approach_y,
        discovered_step=current.step,
        confirmed_step=current.step,
    )
    return reverse


def _connector_id(source_map: str, side: str | None, source_x: int | None, source_y: int | None) -> str:
    if side is not None:
        return f"{source_map}::side::{side}"
    return f"{source_map}::tile::{source_x}:{source_y}"


def _merge_coordinates(
    existing: list[WorldCoordinate],
    updates: Iterable[WorldCoordinate],
) -> list[WorldCoordinate]:
    merged = {(coordinate.x, coordinate.y): coordinate for coordinate in existing}
    for coordinate in updates:
        merged[(coordinate.x, coordinate.y)] = coordinate
    return [merged[key] for key in sorted(merged.keys(), key=lambda item: (item[1], item[0]))]


def _boundary_side(navigation: NavigationSnapshot, x: int, y: int) -> str | None:
    return visible_boundary_side(navigation, x, y)


def _adjacent_walkable_count(x: int, y: int, walkable: set[tuple[int, int]]) -> int:
    count = 0
    for dx, dy in MOVE_DELTAS.values():
        if (x + dx, y + dy) in walkable:
            count += 1
    return count


def _blocked_neighbor_count(x: int, y: int, blocked: set[tuple[int, int]]) -> int:
    count = 0
    for dx, dy in MOVE_DELTAS.values():
        if (x + dx, y + dy) in blocked:
            count += 1
    return count


def _kind_for_transition(side: str | None, source_x: int | None, source_y: int | None) -> str:
    if side is not None:
        return "boundary"
    if source_x is not None and source_y is not None:
        return "door"
    return "unknown"


def _kind_rank(kind: str) -> int:
    ranks = {
        "unknown": 0,
        "boundary": 1,
        "door": 2,
        "stairs": 3,
        "warp": 4,
    }
    return ranks.get(kind, 0)
