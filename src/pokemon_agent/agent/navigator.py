from __future__ import annotations

from dataclasses import dataclass

from pokemon_agent.agent.navigation import NavigationGrid
from pokemon_agent.agent.navigation import advance_position
from pokemon_agent.agent.navigation import is_real_map_edge
from pokemon_agent.agent.navigation import visible_boundary_side
from pokemon_agent.agent.world_map import connectors_from_map
from pokemon_agent.agent.world_map import shortest_confirmed_path
from pokemon_agent.models.action import ActionType
from pokemon_agent.models.memory import DiscoveredConnector
from pokemon_agent.models.memory import WorldMapMemory
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.navigation.world_graph import WorldGraph
from pokemon_agent.navigation.world_graph import WorldGraphEdge
from pokemon_agent.navigation.world_graph import map_matches


@dataclass(frozen=True, slots=True)
class ConnectorApproach:
    x: int
    y: int
    transition_action: ActionType


@dataclass(frozen=True, slots=True)
class RouteStep:
    kind: str
    destination_map: str | None = None
    direction: str | None = None
    connector: DiscoveredConnector | None = None
    edge: WorldGraphEdge | None = None


class Navigator:
    def __init__(
        self,
        world_graph: WorldGraph,
        world_map: WorldMapMemory,
        *,
        blocked_ttl: int = 6,
    ) -> None:
        self._world_graph = world_graph
        self._world_map = world_map
        self._blocked_ttl = max(1, int(blocked_ttl))
        self._blocked_tiles: dict[tuple[str | int, int, int], int] = {}
        self._grid = NavigationGrid()
        self._current_map_key: str | int | None = None

    def update(self, state: StructuredGameState, *, turn_index: int | None = None) -> None:
        if turn_index is not None:
            self.prune_blocked(turn_index)
        self._current_map_key = self.state_map_key(state)
        self._grid = NavigationGrid(state.navigation)
        for blocked_x, blocked_y in self.blocked_tiles_for_state(state):
            self._grid.mark_blocked(blocked_x, blocked_y)

    def prune_blocked(self, turn_index: int) -> None:
        self._blocked_tiles = {
            key: expires_at
            for key, expires_at in self._blocked_tiles.items()
            if expires_at >= turn_index
        }

    def mark_blocked(self, state: StructuredGameState, x: int, y: int, *, turn_index: int) -> None:
        map_key = self.state_map_key(state)
        if map_key is None:
            return
        self._blocked_tiles[(map_key, x, y)] = turn_index + self._blocked_ttl
        if self._current_map_key == map_key:
            self._grid.mark_blocked(x, y)

    def clear_map(self, map_ref: str | int | None) -> None:
        map_key = self._normalize_map_key(map_ref)
        if map_key is None:
            return
        self._blocked_tiles = {
            key: expires_at
            for key, expires_at in self._blocked_tiles.items()
            if key[0] != map_key
        }
        if self._current_map_key == map_key:
            self._grid.clear_blocked()

    def blocked_tiles_for_state(self, state: StructuredGameState) -> set[tuple[int, int]]:
        map_key = self.state_map_key(state)
        if map_key is None:
            return set()
        return {
            (x, y)
            for candidate_map_key, x, y in self._blocked_tiles
            if candidate_map_key == map_key
        }

    def find_local_path(
        self,
        state: StructuredGameState,
        target_x: int,
        target_y: int,
    ) -> list[ActionType] | None:
        if state.x is None or state.y is None:
            return None
        return self._grid.find_path(state.x, state.y, target_x, target_y)

    def best_adjacent_tile(self, target_x: int, target_y: int) -> tuple[int, int] | None:
        return self._grid.best_adjacent_tile(target_x, target_y)

    def best_boundary_target(
        self,
        state: StructuredGameState,
        direction: str,
    ) -> tuple[int, int] | None:
        navigation = self._grid.snapshot
        if navigation is None or state.x is None or state.y is None:
            return None
        best: tuple[tuple[int, int, int], tuple[int, int]] | None = None
        for coordinate in navigation.walkable:
            if (coordinate.x, coordinate.y) == (state.x, state.y):
                continue
            visible_side = visible_boundary_side(navigation, coordinate.x, coordinate.y)
            if visible_side is None:
                continue
            if visible_side != direction and not self._moves_toward_side(
                state.x,
                state.y,
                coordinate.x,
                coordinate.y,
                direction,
            ):
                continue
            route = self.find_local_path(state, coordinate.x, coordinate.y)
            if route is None:
                continue
            rank = self._directional_rank(
                direction,
                coordinate.x,
                coordinate.y,
                len(route),
                is_real_map_edge(navigation, visible_side),
            )
            if best is None or rank < best[0]:
                best = (rank, (coordinate.x, coordinate.y))
        return None if best is None else best[1]

    def approach_for_connector(
        self,
        state: StructuredGameState,
        connector: DiscoveredConnector,
    ) -> ConnectorApproach | None:
        if (
            connector.approach_x is not None
            and connector.approach_y is not None
            and connector.transition_action is not None
        ):
            route = self.find_local_path(state, connector.approach_x, connector.approach_y)
            if route is not None:
                return ConnectorApproach(
                    x=connector.approach_x,
                    y=connector.approach_y,
                    transition_action=connector.transition_action,
                )
        if connector.source_x is None or connector.source_y is None:
            return None
        return self.approach_for_tile(state, connector.source_x, connector.source_y)

    def approach_for_tile(
        self,
        state: StructuredGameState,
        source_x: int,
        source_y: int,
    ) -> ConnectorApproach | None:
        if state.x is None or state.y is None:
            return None
        best: tuple[tuple[int, int, str], ConnectorApproach] | None = None
        source_is_walkable = self._grid.is_walkable(source_x, source_y)
        prefer_downward_entry = bool(
            source_is_walkable
            and state.navigation is not None
            and source_y >= state.navigation.max_y
        )
        for action, dx, dy in (
            (ActionType.MOVE_UP, 0, -1),
            (ActionType.MOVE_RIGHT, 1, 0),
            (ActionType.MOVE_DOWN, 0, 1),
            (ActionType.MOVE_LEFT, -1, 0),
        ):
            approach_x = source_x - dx
            approach_y = source_y - dy
            if not self._grid.is_walkable(approach_x, approach_y):
                continue
            route = self.find_local_path(state, approach_x, approach_y)
            if route is None:
                continue
            direction_penalty = 100 if prefer_downward_entry and action != ActionType.MOVE_DOWN else 0
            rank = (len(route) + direction_penalty, direction_penalty, action.value)
            candidate = ConnectorApproach(
                x=approach_x,
                y=approach_y,
                transition_action=action,
            )
            if best is None or rank < best[0]:
                best = (rank, candidate)

        if best is not None:
            return best[1]

        if source_is_walkable:
            route = self.find_local_path(state, source_x, source_y)
            if route:
                return ConnectorApproach(
                    x=source_x,
                    y=source_y,
                    transition_action=route[-1],
                )
        return None

    def next_route_step(
        self,
        from_map: str | None,
        to_map: str | None,
        *,
        preferred_connector_id: str | None = None,
    ) -> RouteStep | None:
        if not from_map or not to_map or map_matches(from_map, to_map):
            return None

        if preferred_connector_id:
            preferred = self._world_map.connectors.get(preferred_connector_id)
            if (
                preferred is not None
                and preferred.source_map == from_map
                and preferred.destination_map is not None
                and map_matches(preferred.destination_map, to_map)
            ):
                return RouteStep(
                    kind=preferred.kind,
                    destination_map=preferred.destination_map,
                    direction=preferred.source_side,
                    connector=preferred,
                )

        confirmed = shortest_confirmed_path(self._world_map, from_map, to_map)
        if confirmed:
            connector = confirmed[0]
            return RouteStep(
                kind=connector.kind,
                destination_map=connector.destination_map,
                direction=connector.source_side,
                connector=connector,
            )

        route = self._world_graph.find_route(from_map, to_map)
        if route is None or not route.edges:
            return None
        edge = route.edges[0]
        connector = self._connector_for_edge(from_map, edge)
        if connector is not None:
            return RouteStep(
                kind=connector.kind,
                destination_map=connector.destination_map,
                direction=connector.source_side,
                connector=connector,
                edge=edge,
            )
        return RouteStep(
            kind=edge.kind,
            destination_map=edge.destination_name,
            direction=edge.direction,
            edge=edge,
        )

    def coordinates_for_route(
        self,
        state: StructuredGameState,
        route: list[ActionType],
    ) -> list[tuple[int, int]]:
        if state.x is None or state.y is None:
            return []
        coordinates: list[tuple[int, int]] = []
        current_x = state.x
        current_y = state.y
        for action in route:
            coordinate = advance_position(current_x, current_y, action)
            coordinates.append((coordinate.x, coordinate.y))
            current_x = coordinate.x
            current_y = coordinate.y
        return coordinates

    def export_state(self) -> dict[str, object] | None:
        if not self._blocked_tiles:
            return None
        blocked_tiles = [
            {"map_key": key[0], "x": key[1], "y": key[2], "expires_at": expires_at}
            for key, expires_at in sorted(
                self._blocked_tiles.items(),
                key=lambda item: (str(item[0][0]), item[0][2], item[0][1]),
            )
        ]
        return {"blocked_tiles": blocked_tiles}

    def restore_state(self, payload: dict[str, object] | None) -> None:
        self._blocked_tiles = {}
        if not payload:
            return
        for item in payload.get("blocked_tiles", []):
            if not isinstance(item, dict):
                continue
            map_key = self._normalize_map_key(item.get("map_key"))
            x = item.get("x")
            y = item.get("y")
            expires_at = item.get("expires_at")
            if map_key is None or not isinstance(x, int) or not isinstance(y, int) or not isinstance(expires_at, int):
                continue
            self._blocked_tiles[(map_key, x, y)] = expires_at

    @staticmethod
    def state_map_key(state: StructuredGameState) -> str | int | None:
        return state.map_id if state.map_id is not None else (state.map_name or None)

    @staticmethod
    def action_for_direction(direction: str | None) -> ActionType | None:
        mapping = {
            "north": ActionType.MOVE_UP,
            "east": ActionType.MOVE_RIGHT,
            "south": ActionType.MOVE_DOWN,
            "west": ActionType.MOVE_LEFT,
        }
        return mapping.get(direction or "")

    @staticmethod
    def _normalize_map_key(map_ref: object) -> str | int | None:
        if isinstance(map_ref, (str, int)):
            return map_ref
        return None

    def _connector_for_edge(
        self,
        from_map: str,
        edge: WorldGraphEdge,
    ) -> DiscoveredConnector | None:
        ranked: list[tuple[tuple[int, int, int, str], DiscoveredConnector]] = []
        for connector in connectors_from_map(self._world_map, from_map, confirmed_only=False):
            destination = connector.destination_map
            destination_matches = bool(
                destination is not None
                and edge.destination_name is not None
                and map_matches(destination, edge.destination_name)
            )
            if edge.kind == "connection":
                if connector.source_side != edge.direction:
                    continue
            elif edge.kind == "warp":
                if edge.x is not None and edge.y is not None:
                    if connector.source_x != edge.x or connector.source_y != edge.y:
                        continue
                elif not destination_matches:
                    continue
            if edge.kind == "connection" and edge.destination_name is not None and not destination_matches:
                continue
            rank = (
                0 if connector.status.value == "confirmed" else 1,
                0 if destination_matches else 1,
                0 if connector.approach_x is not None and connector.approach_y is not None else 1,
                connector.id,
            )
            ranked.append((rank, connector))
        ranked.sort(key=lambda item: item[0])
        return None if not ranked else ranked[0][1]

    @staticmethod
    def _moves_toward_side(
        current_x: int,
        current_y: int,
        target_x: int,
        target_y: int,
        direction: str,
    ) -> bool:
        if direction == "north":
            return target_y < current_y
        if direction == "east":
            return target_x > current_x
        if direction == "south":
            return target_y > current_y
        if direction == "west":
            return target_x < current_x
        return False

    @staticmethod
    def _directional_rank(
        direction: str,
        x: int,
        y: int,
        distance: int,
        at_real_edge: bool,
    ) -> tuple[int, int, int]:
        if direction == "north":
            return (0 if at_real_edge else 1, y, distance)
        if direction == "east":
            return (0 if at_real_edge else 1, -x, distance)
        if direction == "south":
            return (0 if at_real_edge else 1, -y, distance)
        if direction == "west":
            return (0 if at_real_edge else 1, x, distance)
        return (0, distance, x + y)
