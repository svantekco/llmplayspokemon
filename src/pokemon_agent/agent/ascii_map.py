from __future__ import annotations

from pokemon_agent.emulator.screen_renderer import render_ascii_map
from pokemon_agent.models.state import StructuredGameState


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
