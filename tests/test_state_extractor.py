from __future__ import annotations

import numpy as np
from pokemon_agent.emulator.pokemon_red_ram_map import BAG_ITEMS_ADDR
from pokemon_agent.emulator.pokemon_red_ram_map import EVENT_FLAGS_ADDR
from pokemon_agent.emulator.pokemon_red_ram_map import FIELD_MAP
from pokemon_agent.emulator.pokemon_red_ram_map import MON_HP_OFFSET
from pokemon_agent.emulator.pokemon_red_ram_map import MON_LEVEL_OFFSET
from pokemon_agent.emulator.pokemon_red_ram_map import MON_MAX_HP_OFFSET
from pokemon_agent.emulator.pokemon_red_ram_map import MON_STATUS_OFFSET
from pokemon_agent.emulator.pokemon_red_ram_map import PARTY_MONS_ADDR
from pokemon_agent.emulator.state_extractor import PokemonRedStateExtractor
from pokemon_agent.emulator.pokemon_red_symbol_tables import WATCHED_EVENT_FLAGS
from pokemon_agent.models.state import GameMode


class _FakeMemory(dict):
    def __getitem__(self, key):
        return dict.get(self, key, 0)


class _FakeWrapper:
    def __init__(self, game_area, collision_area=None) -> None:
        self._game_area = game_area
        self._collision_area = collision_area if collision_area is not None else np.zeros_like(game_area)

    def game_area(self):
        return self._game_area

    def game_area_collision(self):
        return self._collision_area


class _FakePyBoy:
    def __init__(self, game_area, memory=None) -> None:
        self.memory = _FakeMemory(memory or {})
        self.game_wrapper = _FakeWrapper(game_area)


def _encode_text_box(lines: list[str], *, yes_no_prompt: bool = False) -> np.ndarray:
    encode = {
        " ": 0x7F,
        "'": 0xE0,
        "!": 0xE7,
        "?": 0xE6,
    }
    encode.update({letter: 0x80 + index for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
    encode.update({letter: 0xA0 + index for index, letter in enumerate("abcdefghijklmnopqrstuvwxyz")})

    grid = np.zeros((18, 20), dtype=np.uint32)
    grid[12, 0] = 0x79
    grid[12, 19] = 0x7B
    grid[17, 0] = 0x7D
    grid[17, 19] = 0x7E
    grid[12, 1:19] = 0x7A
    grid[17, 1:19] = 0x7A
    for row in range(13, 17):
        grid[row, 0] = 0x7C
        grid[row, 19] = 0x7C

    for row_index, line in enumerate(lines, start=13):
        for col_index, char in enumerate(line[:18], start=1):
            grid[row_index, col_index] = encode[char]

    if yes_no_prompt:
        grid[14, 12] = 0xED
        for offset, char in enumerate("YES", start=13):
            grid[14, offset] = encode[char]
        for offset, char in enumerate("NO", start=13):
            grid[15, offset] = encode[char]
    return grid


def test_state_extractor_marks_title_screen_as_bootstrap():
    game_area = np.full((18, 20), 383, dtype=np.uint32)
    game_area[1][2] = 128
    game_area[1][17] = 143
    game_area[7][2] = 305
    game_area[7][17] = 320
    game_area[17][2] = 321
    game_area[17][17] = 334
    extractor = PokemonRedStateExtractor(_FakePyBoy(game_area))

    state = extractor.extract(step=1800)

    assert state.is_bootstrap() is True
    assert state.bootstrap_phase() == "title_screen"
    assert state.map_name == "Title Screen"
    assert state.mode == GameMode.CUTSCENE
    assert state.x is None
    assert state.y is None


def test_state_extractor_marks_title_menu_as_bootstrap():
    game_area = np.full((18, 20), 383, dtype=np.uint32)
    game_area[0][0] = 377
    game_area[0][14] = 379
    game_area[5][0] = 381
    game_area[5][14] = 382
    game_area[2][1] = 237
    extractor = PokemonRedStateExtractor(_FakePyBoy(game_area))

    state = extractor.extract(step=2000)

    assert state.is_bootstrap() is True
    assert state.bootstrap_phase() == "title_menu"
    assert state.menu_open is True


def test_state_extractor_marks_early_zero_coords_as_bootstrap():
    game_area = np.full((18, 20), 383, dtype=np.uint32)
    extractor = PokemonRedStateExtractor(_FakePyBoy(game_area))

    state = extractor.extract(step=500)

    assert state.is_bootstrap() is True
    assert state.bootstrap_phase() == "boot_sequence"


def test_state_extractor_exposes_symbolic_ram_context():
    game_area = np.full((18, 20), 383, dtype=np.uint32)
    collision_area = np.zeros((18, 20), dtype=np.uint32)
    collision_area[6][3] = 1
    collision_area[6][4] = 1
    memory = {
        FIELD_MAP["options"].address: 0x01,
        FIELD_MAP["obtained_badges"].address: 0b00000011,
        FIELD_MAP["bag_count"].address: 1,
        FIELD_MAP["money_1"].address: 0x00,
        FIELD_MAP["money_2"].address: 0x12,
        FIELD_MAP["money_3"].address: 0x34,
        FIELD_MAP["party_count"].address: 2,
        FIELD_MAP["current_map"].address: 0x26,
        FIELD_MAP["player_y"].address: 6,
        FIELD_MAP["player_x"].address: 3,
        FIELD_MAP["player_block_y"].address: 1,
        FIELD_MAP["player_block_x"].address: 0,
        FIELD_MAP["map_tileset"].address: 4,
        FIELD_MAP["map_height"].address: 8,
        FIELD_MAP["map_width"].address: 10,
        FIELD_MAP["is_in_battle"].address: 0,
        FIELD_MAP["player_direction"].address: 0x0C,
        FIELD_MAP["player_moving_direction"].address: 0x03,
        FIELD_MAP["joy_ignore"].address: 0x00,
        FIELD_MAP["top_menu_item_y"].address: 2,
        FIELD_MAP["top_menu_item_x"].address: 13,
        FIELD_MAP["current_menu_item"].address: 1,
        FIELD_MAP["max_menu_item"].address: 4,
        FIELD_MAP["window_y"].address: 0x91,
        PARTY_MONS_ADDR: 0x99,
        PARTY_MONS_ADDR + MON_HP_OFFSET: 35,
        PARTY_MONS_ADDR + MON_HP_OFFSET + 1: 0,
        PARTY_MONS_ADDR + MON_STATUS_OFFSET: 0x08,
        PARTY_MONS_ADDR + MON_LEVEL_OFFSET: 12,
        PARTY_MONS_ADDR + MON_MAX_HP_OFFSET: 39,
        PARTY_MONS_ADDR + MON_MAX_HP_OFFSET + 1: 0,
        PARTY_MONS_ADDR + 0x2C: 0xB0,
        PARTY_MONS_ADDR + 0x2C + MON_HP_OFFSET: 20,
        PARTY_MONS_ADDR + 0x2C + MON_HP_OFFSET + 1: 0,
        PARTY_MONS_ADDR + 0x2C + MON_STATUS_OFFSET: 0x40,
        PARTY_MONS_ADDR + 0x2C + MON_LEVEL_OFFSET: 8,
        PARTY_MONS_ADDR + 0x2C + MON_MAX_HP_OFFSET: 21,
        PARTY_MONS_ADDR + 0x2C + MON_MAX_HP_OFFSET + 1: 0,
        BAG_ITEMS_ADDR: 0x14,
        BAG_ITEMS_ADDR + 1: 3,
        EVENT_FLAGS_ADDR + (WATCHED_EVENT_FLAGS["got_starter"][0] // 8): 1 << (WATCHED_EVENT_FLAGS["got_starter"][0] % 8),
    }
    extractor = PokemonRedStateExtractor(
        _FakePyBoy(game_area, memory=memory),
        rom_profile={"id": "pokemon_red_usa_europe_rev0", "compatible": True},
    )
    extractor.pyboy.game_wrapper = _FakeWrapper(game_area, collision_area=collision_area)

    state = extractor.extract(step=42)

    assert state.is_bootstrap() is False
    assert state.map_name == "Red's House 2F"
    assert state.facing == "RIGHT"
    assert [member.name for member in state.party] == ["Bulbasaur", "Charmander"]
    assert state.party[0].level == 12
    assert state.party[0].status == "PSN"
    assert state.inventory[0].name == "Potion"
    assert state.inventory[0].count == 3
    assert state.story_flags == ["got_starter"]
    assert state.badges == ["Boulder", "Cascade"]
    assert state.metadata["ram_profile"]["compatible"] is True
    assert state.metadata["ram_context"]["player"]["money"] == 1234
    assert state.metadata["ram_context"]["player"]["badges"]["names"] == ["Boulder", "Cascade"]
    assert state.metadata["ram_context"]["map"] == {"tileset": 4, "height": 8, "width": 10, "palette": 0}
    assert state.metadata["ram_context"]["ui"]["top_menu_item_y"] == 2
    assert state.metadata["ram_context"]["ui"]["top_menu_item_x"] == 13
    assert state.metadata["ram_context"]["ui"]["current_menu_item"] == 1
    assert state.metadata["ram_context"]["ui"]["max_menu_item"] == 4
    assert state.metadata["ram_context"]["story"]["watched_flags"]["got_starter"] is True
    assert state.metadata["story_flags"] == ["got_starter"]
    assert state.metadata["badges"] == ["Boulder", "Cascade"]
    assert state.navigation is not None
    assert state.navigation.player.x == 3
    assert state.navigation.player.y == 6
    assert state.navigation.min_x == 0
    assert state.navigation.min_y == 0
    assert len(state.game_area or []) == 18
    assert len((state.game_area or [])[0]) == 20
    assert state.collision_area is not None
    assert state.collision_area[6][3] == 1
    assert any(tile.x == 3 and tile.y == 6 for tile in state.navigation.blocked)


def test_state_extractor_uses_expanded_map_name_table():
    game_area = np.full((18, 20), 383, dtype=np.uint32)
    memory = {
        FIELD_MAP["current_map"].address: 0xC7,
        FIELD_MAP["party_count"].address: 1,
        FIELD_MAP["player_y"].address: 6,
        FIELD_MAP["player_x"].address: 3,
        FIELD_MAP["player_block_y"].address: 1,
        FIELD_MAP["player_block_x"].address: 0,
        FIELD_MAP["map_height"].address: 8,
        FIELD_MAP["map_width"].address: 10,
        FIELD_MAP["is_in_battle"].address: 0,
        FIELD_MAP["joy_ignore"].address: 0,
        FIELD_MAP["window_y"].address: 0x91,
    }
    extractor = PokemonRedStateExtractor(_FakePyBoy(game_area, memory=memory))

    state = extractor.extract(step=5)

    assert state.map_name == "Rocket Hideout B1F"


def test_state_extractor_reads_dialogue_text_and_yes_no_prompt():
    game_area = _encode_text_box(["Buy item?"], yes_no_prompt=True)
    memory = {
        FIELD_MAP["party_count"].address: 1,
        FIELD_MAP["current_map"].address: 0x2A,
        FIELD_MAP["player_y"].address: 6,
        FIELD_MAP["player_x"].address: 3,
        FIELD_MAP["player_block_y"].address: 1,
        FIELD_MAP["player_block_x"].address: 0,
        FIELD_MAP["map_height"].address: 8,
        FIELD_MAP["map_width"].address: 10,
        FIELD_MAP["is_in_battle"].address: 0,
        FIELD_MAP["joy_ignore"].address: 0x01,
        FIELD_MAP["window_y"].address: 0x50,
    }
    extractor = PokemonRedStateExtractor(_FakePyBoy(game_area, memory=memory))

    state = extractor.extract(step=12)

    assert state.text_box_open is True
    assert state.metadata["dialogue_text"] == "Buy item?\nYES\nNO"
    assert state.metadata["yes_no_prompt"] is True


def test_state_extractor_marks_intro_cutscene_with_empty_party_and_blank_background():
    game_area = np.full((18, 20), 383, dtype=np.uint32)
    memory = {
        FIELD_MAP["party_count"].address: 0,
        FIELD_MAP["current_map"].address: 0x26,
        FIELD_MAP["player_y"].address: 6,
        FIELD_MAP["player_x"].address: 3,
        FIELD_MAP["player_block_y"].address: 0,
        FIELD_MAP["player_block_x"].address: 1,
        FIELD_MAP["map_height"].address: 0,
        FIELD_MAP["map_width"].address: 0,
        FIELD_MAP["is_in_battle"].address: 0,
        FIELD_MAP["joy_ignore"].address: 0,
        FIELD_MAP["window_y"].address: 0,
    }
    extractor = PokemonRedStateExtractor(_FakePyBoy(game_area, memory=memory))

    state = extractor.extract(step=1882)

    assert state.is_bootstrap() is True
    assert state.bootstrap_phase() == "intro_cutscene"
    assert state.map_name == "Intro Cutscene"
