from __future__ import annotations

from collections import deque
from typing import Iterable

from pydantic import BaseModel, Field

from pokemon_agent.models.action import ActionType
from pokemon_agent.models.state import NavigationSnapshot
from pokemon_agent.models.state import WorldCoordinate

MOVE_DELTAS: tuple[tuple[ActionType, int, int], ...] = (
    (ActionType.MOVE_UP, 0, -1),
    (ActionType.MOVE_RIGHT, 1, 0),
    (ActionType.MOVE_DOWN, 0, 1),
    (ActionType.MOVE_LEFT, -1, 0),
)


class CachedRoute(BaseModel):
    map_id: str | int | None = None
    target_x: int
    target_y: int
    collision_hash: str | None = None
    remaining_actions: list[ActionType] = Field(default_factory=list)
    expected_start: WorldCoordinate


def build_navigation_snapshot_from_collision(
    collision_area,
    player_x: int | None,
    player_y: int | None,
    map_width_blocks: int | None,
    map_height_blocks: int | None,
    collision_hash: str | None = None,
) -> NavigationSnapshot | None:
    if player_x is None or player_y is None:
        return None

    rows = int(getattr(collision_area, "shape", [0, 0])[0] if collision_area is not None else 0)
    cols = int(getattr(collision_area, "shape", [0, 0])[1] if collision_area is not None else 0)
    if rows <= 0 or cols <= 0:
        return None

    map_width_tiles = max(1, int(map_width_blocks or 0) * 2)
    map_height_tiles = max(1, int(map_height_blocks or 0) * 2)
    max_origin_x = max(0, map_width_tiles - cols)
    max_origin_y = max(0, map_height_tiles - rows)
    origin_x = min(max(player_x - (cols // 2), 0), max_origin_x)
    origin_y = min(max(player_y - (rows // 2), 0), max_origin_y)

    walkable: list[WorldCoordinate] = []
    blocked: list[WorldCoordinate] = []
    for row in range(rows):
        world_y = origin_y + row
        if world_y >= map_height_tiles:
            continue
        for col in range(cols):
            world_x = origin_x + col
            if world_x >= map_width_tiles:
                continue
            coordinate = WorldCoordinate(x=world_x, y=world_y)
            if int(collision_area[row][col]) == 0:
                walkable.append(coordinate)
            else:
                blocked.append(coordinate)

    if not walkable and not blocked:
        return None

    max_x = max(coord.x for coord in [*walkable, *blocked])
    max_y = max(coord.y for coord in [*walkable, *blocked])
    return NavigationSnapshot(
        min_x=origin_x,
        min_y=origin_y,
        max_x=max_x,
        max_y=max_y,
        player=WorldCoordinate(x=player_x, y=player_y),
        walkable=walkable,
        blocked=blocked,
        collision_hash=collision_hash,
    )


def build_navigation_snapshot_from_tiles(
    width: int,
    height: int,
    player_x: int | None,
    player_y: int | None,
    blocked_tiles: Iterable[tuple[int, int]],
    collision_hash: str | None = None,
) -> NavigationSnapshot | None:
    if player_x is None or player_y is None:
        return None

    blocked_set = set(blocked_tiles)
    walkable: list[WorldCoordinate] = []
    blocked: list[WorldCoordinate] = []
    for y in range(max(0, height)):
        for x in range(max(0, width)):
            coordinate = WorldCoordinate(x=x, y=y)
            if (x, y) in blocked_set:
                blocked.append(coordinate)
            else:
                walkable.append(coordinate)

    if not walkable and not blocked:
        return None

    return NavigationSnapshot(
        min_x=0,
        min_y=0,
        max_x=max(0, width - 1),
        max_y=max(0, height - 1),
        player=WorldCoordinate(x=player_x, y=player_y),
        walkable=walkable,
        blocked=blocked,
        collision_hash=collision_hash,
    )


def find_path(
    navigation: NavigationSnapshot | None,
    start_x: int | None,
    start_y: int | None,
    target_x: int,
    target_y: int,
) -> list[ActionType] | None:
    if navigation is None or start_x is None or start_y is None:
        return None

    start = (start_x, start_y)
    target = (target_x, target_y)
    walkable = {(coord.x, coord.y) for coord in navigation.walkable}
    if start not in walkable or target not in walkable:
        return None
    if start == target:
        return []

    frontier = deque([start])
    parents: dict[tuple[int, int], tuple[tuple[int, int], ActionType] | None] = {start: None}
    while frontier:
        current_x, current_y = frontier.popleft()
        if (current_x, current_y) == target:
            break
        for action, dx, dy in MOVE_DELTAS:
            neighbor = (current_x + dx, current_y + dy)
            if neighbor in parents or neighbor not in walkable:
                continue
            parents[neighbor] = ((current_x, current_y), action)
            frontier.append(neighbor)

    if target not in parents:
        return None

    route: list[ActionType] = []
    current = target
    while parents[current] is not None:
        parent, action = parents[current]
        route.append(action)
        current = parent
    route.reverse()
    return route


def advance_position(x: int, y: int, action: ActionType) -> WorldCoordinate:
    for candidate, dx, dy in MOVE_DELTAS:
        if action == candidate:
            return WorldCoordinate(x=x + dx, y=y + dy)
    return WorldCoordinate(x=x, y=y)
