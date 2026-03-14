import numpy as np

from pokemon_agent.agent.navigation import build_navigation_snapshot_from_collision
from pokemon_agent.agent.navigation import find_path
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
