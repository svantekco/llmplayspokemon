import numpy as np

from pokemon_agent.agent.navigation import build_navigation_snapshot_from_collision
from pokemon_agent.agent.navigation import build_navigation_snapshot_from_tiles
from pokemon_agent.agent.navigation import facing_action_for_target
from pokemon_agent.agent.navigation import find_path
from pokemon_agent.agent.navigation import NavigationGrid
from pokemon_agent.models.action import ActionType


def _collision_from_logical_grid(logical_grid: list[list[int]]) -> np.ndarray:
    logical = np.array(logical_grid, dtype=np.uint8)
    return np.kron(logical, np.ones((2, 2), dtype=np.uint8))


def test_build_navigation_snapshot_from_collision_aligns_small_indoor_map() -> None:
    collision_area = _collision_from_logical_grid(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    navigation = build_navigation_snapshot_from_collision(
        collision_area=collision_area,
        player_x=5,
        player_y=4,
        map_width_blocks=4,
        map_height_blocks=4,
        collision_hash="reds-house-2f",
    )

    assert navigation is not None
    walkable = {(coordinate.x, coordinate.y) for coordinate in navigation.walkable}
    assert navigation.coverage == "full_map"
    assert navigation.visible_world_edges == ["north", "east", "south", "west"]
    assert (5, 4) in walkable
    assert (6, 1) in walkable
    assert find_path(navigation, 5, 4, 6, 1) == [
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_RIGHT,
    ]


def test_navigation_grid_matches_find_path_on_same_snapshot() -> None:
    navigation = build_navigation_snapshot_from_tiles(
        width=5,
        height=5,
        player_x=2,
        player_y=4,
        blocked_tiles=[(2, 2)],
        collision_hash="grid",
    )

    assert navigation is not None
    grid = NavigationGrid(navigation)

    assert grid.find_path(2, 4, 2, 0) == find_path(navigation, 2, 4, 2, 0)


def test_navigation_grid_avoids_task_scoped_blocked_tiles() -> None:
    navigation = build_navigation_snapshot_from_tiles(
        width=5,
        height=5,
        player_x=2,
        player_y=4,
        blocked_tiles=[],
        collision_hash="open-grid",
    )

    assert navigation is not None
    grid = NavigationGrid(navigation)

    assert grid.find_path(2, 4, 2, 0) == [
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
    ]

    grid.mark_blocked(2, 3)

    assert grid.find_path(2, 4, 2, 0) == [
        ActionType.MOVE_RIGHT,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_UP,
        ActionType.MOVE_LEFT,
    ]


def test_navigation_grid_best_adjacent_tile_ignores_blocked_options() -> None:
    navigation = build_navigation_snapshot_from_tiles(
        width=4,
        height=4,
        player_x=0,
        player_y=0,
        blocked_tiles=[(1, 0), (0, 1)],
        collision_hash="adjacent-grid",
    )

    assert navigation is not None
    grid = NavigationGrid(navigation)
    grid.mark_blocked(2, 1)

    assert grid.best_adjacent_tile(1, 1) == (1, 2)


def test_facing_action_for_target_covers_all_cardinal_directions() -> None:
    assert facing_action_for_target(3, 3, 3, 2) == ActionType.MOVE_UP
    assert facing_action_for_target(3, 3, 4, 3) == ActionType.MOVE_RIGHT
    assert facing_action_for_target(3, 3, 3, 4) == ActionType.MOVE_DOWN
    assert facing_action_for_target(3, 3, 2, 3) == ActionType.MOVE_LEFT


def test_build_navigation_snapshot_uses_explicit_screen_origin_from_ram() -> None:
    collision_area = _collision_from_logical_grid(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        ]
    )

    navigation = build_navigation_snapshot_from_collision(
        collision_area=collision_area,
        player_x=7,
        player_y=1,
        map_width_blocks=4,
        map_height_blocks=4,
        screen_origin_x=3,
        screen_origin_y=-3,
        collision_hash="reds-house-2f-stairs",
    )

    assert navigation is not None
    walkable = {(coordinate.x, coordinate.y) for coordinate in navigation.walkable}
    assert navigation.screen_origin_x == 3
    assert navigation.screen_origin_y == -3
    assert (7, 1) in walkable
    assert (7, 3) in walkable
