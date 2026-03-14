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

    logical_collision = _logical_collision_grid(collision_area)
    rows = len(logical_collision)
    cols = len(logical_collision[0]) if rows else 0
    if rows <= 0 or cols <= 0:
        return None

    map_width_tiles = max(1, int(map_width_blocks or 0) * 2)
    map_height_tiles = max(1, int(map_height_blocks or 0) * 2)
    origin_x, origin_y = _select_visible_origin(
        logical_collision,
        player_x=player_x,
        player_y=player_y,
        map_width_tiles=map_width_tiles,
        map_height_tiles=map_height_tiles,
    )

    walkable: list[WorldCoordinate] = []
    blocked: list[WorldCoordinate] = []
    for row in range(rows):
        world_y = origin_y + row
        if world_y < 0 or world_y >= map_height_tiles:
            continue
        for col in range(cols):
            world_x = origin_x + col
            if world_x < 0 or world_x >= map_width_tiles:
                continue
            coordinate = WorldCoordinate(x=world_x, y=world_y)
            if logical_collision[row][col] != 0:
                walkable.append(coordinate)
            else:
                blocked.append(coordinate)

    if not walkable and not blocked:
        return None

    player_coordinate = (player_x, player_y)
    blocked = [coordinate for coordinate in blocked if (coordinate.x, coordinate.y) != player_coordinate]
    if player_coordinate not in {(coordinate.x, coordinate.y) for coordinate in walkable}:
        walkable.append(WorldCoordinate(x=player_x, y=player_y))
    walkable.sort(key=lambda item: (item.y, item.x))
    blocked.sort(key=lambda item: (item.y, item.x))

    all_coordinates = [*walkable, *blocked]
    min_x = min(coord.x for coord in all_coordinates)
    min_y = min(coord.y for coord in all_coordinates)
    max_x = max(coord.x for coord in all_coordinates)
    max_y = max(coord.y for coord in all_coordinates)
    visible_world_edges: list[str] = []
    if origin_y <= 0:
        visible_world_edges.append("north")
    if origin_x + cols >= map_width_tiles:
        visible_world_edges.append("east")
    if origin_y + rows >= map_height_tiles:
        visible_world_edges.append("south")
    if origin_x <= 0:
        visible_world_edges.append("west")
    coverage = "full_map" if map_width_tiles <= cols and map_height_tiles <= rows else "local_window"

    return NavigationSnapshot(
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        player=WorldCoordinate(x=player_x, y=player_y),
        walkable=walkable,
        blocked=blocked,
        collision_hash=collision_hash,
        coverage=coverage,
        map_width=map_width_tiles,
        map_height=map_height_tiles,
        visible_world_edges=visible_world_edges,
        screen_origin_x=origin_x,
        screen_origin_y=origin_y,
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
        coverage="full_map",
        map_width=width,
        map_height=height,
        visible_world_edges=["north", "east", "south", "west"],
        screen_origin_x=0,
        screen_origin_y=0,
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


def visible_boundary_side(navigation: NavigationSnapshot, x: int, y: int) -> str | None:
    if y == navigation.min_y:
        return "north"
    if x == navigation.max_x:
        return "east"
    if y == navigation.max_y:
        return "south"
    if x == navigation.min_x:
        return "west"
    return None


def is_real_map_edge(navigation: NavigationSnapshot, side: str) -> bool:
    if navigation.coverage == "full_map" and not navigation.visible_world_edges:
        return True
    return side in navigation.visible_world_edges


def _logical_collision_grid(collision_area) -> list[list[int]]:
    rows = int(getattr(collision_area, "shape", [0, 0])[0] if collision_area is not None else 0)
    cols = int(getattr(collision_area, "shape", [0, 0])[1] if collision_area is not None else 0)
    if rows <= 0 or cols <= 0:
        return []
    logical_rows = (rows + 1) // 2
    logical_cols = (cols + 1) // 2
    logical: list[list[int]] = []
    for logical_row in range(logical_rows):
        row_values: list[int] = []
        base_row = logical_row * 2
        for logical_col in range(logical_cols):
            base_col = logical_col * 2
            walkable = False
            for row_offset in (0, 1):
                row_index = base_row + row_offset
                if row_index >= rows:
                    continue
                for col_offset in (0, 1):
                    col_index = base_col + col_offset
                    if col_index >= cols:
                        continue
                    if int(collision_area[row_index][col_index]) != 0:
                        walkable = True
                        break
                if walkable:
                    break
            row_values.append(1 if walkable else 0)
        logical.append(row_values)
    return logical


def _select_visible_origin(
    logical_collision: list[list[int]],
    *,
    player_x: int,
    player_y: int,
    map_width_tiles: int,
    map_height_tiles: int,
) -> tuple[int, int]:
    rows = len(logical_collision)
    cols = len(logical_collision[0]) if rows else 0
    min_origin_x = min(0, map_width_tiles - cols)
    max_origin_x = max(0, map_width_tiles - cols)
    min_origin_y = min(0, map_height_tiles - rows)
    max_origin_y = max(0, map_height_tiles - rows)
    expected_origin_x = min(max(player_x - (cols // 2), min_origin_x), max_origin_x)
    expected_origin_y = min(max(player_y - (rows // 2), min_origin_y), max_origin_y)

    best_rank: tuple[int, int, int, int, int, int] | None = None
    best_origin = (expected_origin_x, expected_origin_y)
    for origin_y in range(min_origin_y, max_origin_y + 1):
        for origin_x in range(min_origin_x, max_origin_x + 1):
            screen_x = player_x - origin_x
            screen_y = player_y - origin_y
            if not (0 <= screen_x < cols and 0 <= screen_y < rows):
                continue
            player_walkable = logical_collision[screen_y][screen_x] != 0
            outside_blocked = 0
            outside_walkable = 0
            inside_walkable = 0
            inside_blocked = 0
            for row_index, row_values in enumerate(logical_collision):
                world_y = origin_y + row_index
                for col_index, value in enumerate(row_values):
                    world_x = origin_x + col_index
                    inside_map = 0 <= world_x < map_width_tiles and 0 <= world_y < map_height_tiles
                    if inside_map:
                        if value != 0:
                            inside_walkable += 1
                        else:
                            inside_blocked += 1
                    elif value != 0:
                        outside_walkable += 1
                    else:
                        outside_blocked += 1
            center_distance = abs(origin_x - expected_origin_x) + abs(origin_y - expected_origin_y)
            rank = (
                1 if player_walkable else 0,
                outside_blocked - outside_walkable,
                inside_walkable - inside_blocked,
                -center_distance,
                -abs(origin_y),
                -abs(origin_x),
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_origin = (origin_x, origin_y)
    return best_origin
