from pokemon_agent.emulator.screen_renderer import render_ascii_map


def test_render_ascii_map_marks_special_tiles():
    game_area = [[0 for _ in range(20)] for _ in range(18)]
    collision_area = [[0 for _ in range(20)] for _ in range(18)]

    for row in (4, 5):
        for col in (4, 5):
            game_area[row][col] = 77
            collision_area[row][col] = 1

    collision_area[8][11] = 1
    collision_area[9][0] = 1
    collision_area[10][14] = 1
    collision_area[10][15] = 1

    rendered = render_ascii_map(game_area, collision_area, player_x=10, player_y=8)
    lines = rendered.splitlines()

    assert len(lines) == 18
    assert all(len(line) == 20 for line in lines)
    assert lines[4][4] == "~"
    assert lines[5][5] == "~"
    assert lines[8][10] == "P"
    assert lines[8][11] == "@"
    assert lines[9][0] == "D"
    assert lines[10][14] == "#"
