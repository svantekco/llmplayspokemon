from __future__ import annotations

from pokemon_agent.emulator.pokemon_red_ram_map import BAG_ITEMS_ADDR
from pokemon_agent.emulator.pokemon_red_ram_map import FIELD_MAP
from pokemon_agent.emulator.pokemon_red_ram_map import MON_HP_OFFSET
from pokemon_agent.emulator.pokemon_red_ram_map import MON_LEVEL_OFFSET
from pokemon_agent.emulator.pokemon_red_ram_map import MON_MAX_HP_OFFSET
from pokemon_agent.emulator.pokemon_red_ram_map import MON_STATUS_OFFSET
from pokemon_agent.emulator.pokemon_red_ram_map import NUM_BAG_ITEMS_ADDR
from pokemon_agent.emulator.pokemon_red_ram_map import PARTY_MONS_ADDR
from pokemon_agent.emulator.pokemon_red_ram_map import PARTYMON_STRUCT_LENGTH
from pokemon_agent.emulator.pokemon_red_ram_map import WATCHED_EVENT_FLAGS
from pokemon_agent.emulator.pokemon_red_ram_map import build_ram_context
from pokemon_agent.emulator.pokemon_red_ram_map import verify_pokemon_red_rom


class _FakeMemory(dict):
    def __getitem__(self, key):
        return dict.get(self, key, 0)


def _build_header_rom() -> bytes:
    rom = bytearray(0x150)
    rom[0x134:0x144] = b"POKEMON RED\x00\x00\x00\x00\x00"
    rom[0x143] = 0x00
    rom[0x146] = 0x03
    rom[0x147] = 0x13
    rom[0x148] = 0x05
    rom[0x149] = 0x03
    rom[0x14A] = 0x01
    rom[0x14B] = 0x33
    rom[0x14C] = 0x00
    rom[0x14D] = 0x20
    rom[0x14E:0x150] = bytes.fromhex("91e6")
    return bytes(rom)


def test_verify_pokemon_red_rom_accepts_matching_header():
    verification = verify_pokemon_red_rom(_build_header_rom())

    assert verification.compatible is True
    assert verification.header.title == "POKEMON RED"
    assert verification.header.version == 0
    assert verification.exact_sha1_match is False
    assert "title" in verification.matched_fields


def test_build_ram_context_decodes_player_map_ui_and_battle_fields():
    memory = _FakeMemory(
        {
            FIELD_MAP["options"].address: 0x41,
            FIELD_MAP["obtained_badges"].address: 0b00000101,
            FIELD_MAP["bag_count"].address: 2,
            FIELD_MAP["money_1"].address: 0x12,
            FIELD_MAP["money_2"].address: 0x34,
            FIELD_MAP["money_3"].address: 0x56,
            FIELD_MAP["party_count"].address: 3,
            FIELD_MAP["map_palette"].address: 6,
            FIELD_MAP["current_map"].address: 0x26,
            FIELD_MAP["player_y"].address: 6,
            FIELD_MAP["player_x"].address: 3,
            FIELD_MAP["player_block_y"].address: 1,
            FIELD_MAP["player_block_x"].address: 0,
            FIELD_MAP["map_tileset"].address: 4,
            FIELD_MAP["map_height"].address: 8,
            FIELD_MAP["map_width"].address: 10,
            FIELD_MAP["is_in_battle"].address: 2,
            FIELD_MAP["player_moving_direction"].address: 3,
            FIELD_MAP["player_direction"].address: 0x0C,
            FIELD_MAP["joy_ignore"].address: 0x40,
            FIELD_MAP["top_menu_item_y"].address: 2,
            FIELD_MAP["top_menu_item_x"].address: 13,
            FIELD_MAP["current_menu_item"].address: 1,
            FIELD_MAP["max_menu_item"].address: 4,
            FIELD_MAP["window_y"].address: 0x70,
            NUM_BAG_ITEMS_ADDR: 2,
            BAG_ITEMS_ADDR: 0x14,
            BAG_ITEMS_ADDR + 1: 3,
            BAG_ITEMS_ADDR + 2: 0x4A,
            BAG_ITEMS_ADDR + 3: 1,
            PARTY_MONS_ADDR: 0x99,
            PARTY_MONS_ADDR + MON_HP_OFFSET: 35,
            PARTY_MONS_ADDR + MON_HP_OFFSET + 1: 0,
            PARTY_MONS_ADDR + MON_STATUS_OFFSET: 0x08,
            PARTY_MONS_ADDR + MON_LEVEL_OFFSET: 12,
            PARTY_MONS_ADDR + MON_MAX_HP_OFFSET: 39,
            PARTY_MONS_ADDR + MON_MAX_HP_OFFSET + 1: 0,
            PARTY_MONS_ADDR + PARTYMON_STRUCT_LENGTH: 0xB0,
            PARTY_MONS_ADDR + PARTYMON_STRUCT_LENGTH + MON_HP_OFFSET: 20,
            PARTY_MONS_ADDR + PARTYMON_STRUCT_LENGTH + MON_HP_OFFSET + 1: 0,
            PARTY_MONS_ADDR + PARTYMON_STRUCT_LENGTH + MON_STATUS_OFFSET: 0x40,
            PARTY_MONS_ADDR + PARTYMON_STRUCT_LENGTH + MON_LEVEL_OFFSET: 8,
            PARTY_MONS_ADDR + PARTYMON_STRUCT_LENGTH + MON_MAX_HP_OFFSET: 21,
            PARTY_MONS_ADDR + PARTYMON_STRUCT_LENGTH + MON_MAX_HP_OFFSET + 1: 0,
            (0xD747 + (WATCHED_EVENT_FLAGS["got_starter"][0] // 8)): 1 << (WATCHED_EVENT_FLAGS["got_starter"][0] % 8),
            (0xD747 + (WATCHED_EVENT_FLAGS["beat_brock"][0] // 8)): 1 << (WATCHED_EVENT_FLAGS["beat_brock"][0] % 8),
        }
    )

    context = build_ram_context(memory)

    assert context["player"]["map_id"] == 0x26
    assert context["player"]["money"] == 123456
    assert context["player"]["party_count"] == 3
    assert context["player"]["bag_count"] == 2
    assert context["player"]["badges"]["count"] == 2
    assert context["player"]["badges"]["names"] == ["Boulder", "Thunder"]
    assert context["party"] == [
        {"name": "Bulbasaur", "species_id": 0x99, "level": 12, "hp": 35, "max_hp": 39, "status": "PSN"},
        {"name": "Charmander", "species_id": 0xB0, "level": 8, "hp": 20, "max_hp": 21, "status": "PAR"},
    ]
    assert context["inventory"] == [
        {"name": "Potion", "item_id": 0x14, "count": 3},
        {"name": "Lift Key", "item_id": 0x4A, "count": 1},
    ]
    assert context["map"] == {
        "tileset": 4,
        "height": 8,
        "width": 10,
        "palette": 6,
        "view_pointer": 0,
        "screen_origin_x": None,
        "screen_origin_y": None,
    }
    assert context["ui"]["input_locked"] is True
    assert context["ui"]["top_menu_item_y"] == 2
    assert context["ui"]["top_menu_item_x"] == 13
    assert context["ui"]["current_menu_item"] == 1
    assert context["ui"]["max_menu_item"] == 4
    assert context["ui"]["options"]["battle_style"] == "SET"
    assert context["ui"]["options"]["text_speed_preset"] == "FAST"
    assert context["battle"] == {"flag": 2, "in_battle": True, "kind": "TRAINER"}
    assert context["story"]["watched_flags"]["got_starter"] is True
    assert context["story"]["watched_flags"]["beat_brock"] is True
    assert context["story"]["active_flags"] == ["got_starter", "beat_brock"]


def test_build_ram_context_derives_screen_origin_from_map_view_pointer():
    memory = _FakeMemory(
        {
            FIELD_MAP["current_map"].address: 0x26,
            FIELD_MAP["player_y"].address: 1,
            FIELD_MAP["player_x"].address: 7,
            FIELD_MAP["player_block_y"].address: 1,
            FIELD_MAP["player_block_x"].address: 1,
            FIELD_MAP["current_map_view_pointer_lo"].address: 0xF6,
            FIELD_MAP["current_map_view_pointer_hi"].address: 0xC6,
            FIELD_MAP["map_height"].address: 4,
            FIELD_MAP["map_width"].address: 4,
        }
    )

    context = build_ram_context(memory)

    assert context["map"]["view_pointer"] == 0xC6F6
    assert context["map"]["screen_origin_x"] == 3
    assert context["map"]["screen_origin_y"] == -3
