from __future__ import annotations

from collections import deque

from pokemon_agent.models.state import StructuredGameState

SCREEN_HEIGHT = 18
SCREEN_WIDTH = 20
ORTHOGONAL_DELTAS: tuple[tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


def build_ascii_map(state: StructuredGameState) -> str | None:
    if state.x is not None and state.y is not None and state.game_area is not None and state.collision_area is not None:
        return render_ascii_map(state.game_area, state.collision_area, state.x, state.y)

    navigation = state.navigation
    if navigation is None or state.x is None or state.y is None:
        return None

    walkable = {(coord.x, coord.y) for coord in navigation.walkable}
    blocked = {(coord.x, coord.y) for coord in navigation.blocked}
    lines: list[str] = []
    for y in range(navigation.min_y, navigation.max_y + 1):
        chars: list[str] = []
        for x in range(navigation.min_x, navigation.max_x + 1):
            if (x, y) == (state.x, state.y):
                chars.append("P")
            elif (x, y) in blocked:
                chars.append("#")
            elif (x, y) in walkable:
                chars.append(".")
            else:
                chars.append(" ")
        lines.append("".join(chars).rstrip())
    return "\n".join(line for line in lines if line)


def render_ascii_map(game_area, collision_area, player_x, player_y) -> str:
    rows = _grid_rows(game_area, collision_area)
    cols = _grid_cols(game_area, collision_area)
    blocked = [[_is_blocked(collision_area, row, col) for col in range(cols)] for row in range(rows)]
    water_components = _water_components(game_area, blocked)

    output_rows: list[str] = []
    for row in range(rows):
        chars: list[str] = []
        for col in range(cols):
            if player_x == col and player_y == row:
                chars.append("P")
                continue
            if not blocked[row][col]:
                chars.append(".")
                continue
            if _looks_like_door(row, col, blocked):
                chars.append("D")
                continue
            if (row, col) in water_components:
                chars.append("~")
                continue
            if _blocked_neighbor_count(row, col, blocked) == 0:
                chars.append("@")
                continue
            chars.append("#")
        output_rows.append("".join(chars))
    return "\n".join(output_rows)


def _grid_rows(game_area, collision_area) -> int:
    return max(_shape_dim(game_area, 0), _shape_dim(collision_area, 0), SCREEN_HEIGHT)


def _grid_cols(game_area, collision_area) -> int:
    return max(_shape_dim(game_area, 1), _shape_dim(collision_area, 1), SCREEN_WIDTH)


def _shape_dim(grid, dim: int) -> int:
    if grid is None:
        return 0
    shape = getattr(grid, "shape", None)
    if shape is not None and len(shape) > dim:
        return int(shape[dim])
    if not grid:
        return 0
    if dim == 0:
        return len(grid)
    first_row = grid[0]
    return len(first_row) if first_row is not None else 0


def _value_at(grid, row: int, col: int, default: int = 0) -> int:
    if grid is None:
        return default
    try:
        return int(grid[row][col])
    except (IndexError, TypeError, ValueError):
        return default


def _is_blocked(collision_area, row: int, col: int) -> bool:
    return _value_at(collision_area, row, col, default=0) != 0


def _blocked_neighbor_count(row: int, col: int, blocked: list[list[bool]]) -> int:
    count = 0
    rows = len(blocked)
    cols = len(blocked[0]) if rows else 0
    for dx, dy in ORTHOGONAL_DELTAS:
        next_row = row + dy
        next_col = col + dx
        if 0 <= next_row < rows and 0 <= next_col < cols and blocked[next_row][next_col]:
            count += 1
    return count


def _looks_like_door(row: int, col: int, blocked: list[list[bool]]) -> bool:
    rows = len(blocked)
    cols = len(blocked[0]) if rows else 0
    if rows == 0 or cols == 0:
        return False

    if row not in {0, rows - 1} and col not in {0, cols - 1}:
        return False

    walk_up = row > 0 and not blocked[row - 1][col]
    walk_down = row + 1 < rows and not blocked[row + 1][col]
    walk_left = col > 0 and not blocked[row][col - 1]
    walk_right = col + 1 < cols and not blocked[row][col + 1]

    return walk_left or walk_right or walk_up or walk_down


def _water_components(game_area, blocked: list[list[bool]]) -> set[tuple[int, int]]:
    rows = len(blocked)
    cols = len(blocked[0]) if rows else 0
    seen: set[tuple[int, int]] = set()
    water: set[tuple[int, int]] = set()

    for row in range(rows):
        for col in range(cols):
            if not blocked[row][col] or (row, col) in seen:
                continue

            tile_id = _value_at(game_area, row, col, default=-1)
            component: set[tuple[int, int]] = set()
            queue = deque([(row, col)])
            spans_rows = False
            spans_cols = False

            while queue:
                current_row, current_col = queue.popleft()
                if (current_row, current_col) in seen:
                    continue
                seen.add((current_row, current_col))
                if not blocked[current_row][current_col]:
                    continue
                if _value_at(game_area, current_row, current_col, default=-1) != tile_id:
                    continue

                component.add((current_row, current_col))
                for dx, dy in ORTHOGONAL_DELTAS:
                    next_row = current_row + dy
                    next_col = current_col + dx
                    if not (0 <= next_row < rows and 0 <= next_col < cols):
                        continue
                    if not blocked[next_row][next_col]:
                        continue
                    if _value_at(game_area, next_row, next_col, default=-1) != tile_id:
                        continue
                    spans_rows = spans_rows or next_row != row
                    spans_cols = spans_cols or next_col != col
                    queue.append((next_row, next_col))

            if len(component) >= 4 and spans_rows and spans_cols:
                water.update(component)
    return water
