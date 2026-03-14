from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from pokemon_agent.data.pokemon_red_battle_data import MOVE_DATA
from pokemon_agent.emulator.pokemon_red_symbol_tables import ITEM_SYMBOLS
from pokemon_agent.emulator.pokemon_red_symbol_tables import POKEMON_SYMBOLS
from pokemon_agent.emulator.pokemon_red_symbol_tables import WATCHED_EVENT_FLAGS

PROFILE_ID = "pokemon_red_usa_europe_rev0"
PROFILE_NAME = "Pokemon Red (USA/Europe, revision 0)"
EXACT_ROM_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"

BADGE_NAMES = (
    "Boulder",
    "Cascade",
    "Thunder",
    "Rainbow",
    "Soul",
    "Marsh",
    "Volcano",
    "Earth",
)

TEXT_SPEED_PRESETS = {
    1: "FAST",
    3: "MEDIUM",
    5: "SLOW",
}

PARTY_LIMIT = 6
BAG_ITEM_CAPACITY = 20
PARTY_SPECIES_ADDR = 0xD164
PARTY_MONS_ADDR = 0xD16B
PARTYMON_STRUCT_LENGTH = 0x2C
NUM_BAG_ITEMS_ADDR = 0xD31D
BAG_ITEMS_ADDR = 0xD31E
EVENT_FLAGS_ADDR = 0xD747
MON_HP_OFFSET = 0x01
MON_STATUS_OFFSET = 0x04
MON_LEVEL_OFFSET = 0x21
MON_MAX_HP_OFFSET = 0x22
ENEMY_SPECIES_ADDR = 0xCFE5
ENEMY_HP_ADDR = 0xCFE6
ENEMY_LEVEL_ADDR = 0xCFF1
ENEMY_MAX_HP_ADDR = 0xCFFC
# Use the verified pokered WRAM cursor and active battle-mon struct offsets for
# navigation/runtime decisions; some requested raw bytes are still preserved below.
CURRENT_MENU_ITEM_ADDR = 0xCC26
REQUESTED_MOVE_CURSOR_ADDR = 0xCC2B
BATTLE_SAVED_MENU_ITEM_ADDR = 0xCC2D
PLAYER_ACTIVE_SPECIES_ADDR = 0xD014
PLAYER_ACTIVE_HP_ADDR = 0xD015
PLAYER_ACTIVE_MOVES_ADDR = 0xD01C
PLAYER_ACTIVE_LEVEL_ADDR = 0xD022
PLAYER_ACTIVE_MAX_HP_ADDR = 0xD023
PLAYER_ACTIVE_PP_ADDR = 0xD02D

SPECIAL_NAME_OVERRIDES = {
    "NO_MON": "None",
    "NO_ITEM": "None",
    "MR_MIME": "Mr. Mime",
    "FARFETCHD": "Farfetch'd",
    "NIDORAN_M": "Nidoran M",
    "NIDORAN_F": "Nidoran F",
    "S_S_TICKET": "S.S. Ticket",
    "OAKS_PARCEL": "Oak's Parcel",
    "PSYCHIC_M": "Psychic",
    "EXP_ALL": "Exp. All",
    "PP_UP": "PP Up",
    "HP_UP": "HP Up",
}


@dataclass(frozen=True, slots=True)
class RamField:
    key: str
    address: int
    description: str
    category: str


@dataclass(frozen=True, slots=True)
class RomHeader:
    title: str
    cgb_flag: int
    sgb_flag: int
    cart_type: int
    rom_size: int
    ram_size: int
    destination: int
    old_licensee: int
    version: int
    header_checksum: int
    global_checksum: int
    sha1: str


@dataclass(frozen=True, slots=True)
class RomVerification:
    profile_id: str
    profile_name: str
    compatible: bool
    exact_sha1_match: bool
    matched_fields: tuple[str, ...]
    warnings: tuple[str, ...]
    header: RomHeader


FIELD_MAP = {
    "options": RamField("options", 0xD355, "Gameplay options byte", "ui"),
    "obtained_badges": RamField("obtained_badges", 0xD356, "Badge flags", "player"),
    "bag_count": RamField("bag_count", NUM_BAG_ITEMS_ADDR, "Number of items in the bag", "player"),
    "money_1": RamField("money_1", 0xD347, "Money byte 1 (BCD)", "player"),
    "money_2": RamField("money_2", 0xD348, "Money byte 2 (BCD)", "player"),
    "money_3": RamField("money_3", 0xD349, "Money byte 3 (BCD)", "player"),
    "party_count": RamField("party_count", 0xD163, "Pokemon in party", "player"),
    "map_palette": RamField("map_palette", 0xD35D, "Map palette override", "map"),
    "current_map": RamField("current_map", 0xD35E, "Current map number", "map"),
    "player_y": RamField("player_y", 0xD361, "Current player Y position", "player"),
    "player_x": RamField("player_x", 0xD362, "Current player X position", "player"),
    "player_block_y": RamField("player_block_y", 0xD363, "Current player block Y position", "player"),
    "player_block_x": RamField("player_block_x", 0xD364, "Current player block X position", "player"),
    "map_tileset": RamField("map_tileset", 0xD367, "Map tileset id", "map"),
    "map_height": RamField("map_height", 0xD368, "Map height in blocks", "map"),
    "map_width": RamField("map_width", 0xD369, "Map width in blocks", "map"),
    "is_in_battle": RamField("is_in_battle", 0xD057, "Battle flag", "battle"),
    "player_moving_direction": RamField(
        "player_moving_direction",
        0xD528,
        "Moving direction byte",
        "player",
    ),
    "player_direction": RamField("player_direction", 0xD52A, "Facing direction byte", "player"),
    "joy_ignore": RamField("joy_ignore", 0xD730, "Ignored joypad mask", "ui"),
    "top_menu_item_y": RamField("top_menu_item_y", 0xCC24, "Top menu cursor row", "ui"),
    "top_menu_item_x": RamField("top_menu_item_x", 0xCC25, "Top menu cursor column", "ui"),
    "current_menu_item": RamField("current_menu_item", 0xCC26, "Current menu cursor index", "ui"),
    "max_menu_item": RamField("max_menu_item", 0xCC28, "Maximum menu cursor index", "ui"),
    "window_y": RamField("window_y", 0xFF4A, "Window Y hardware register", "ui"),
}


def read_rom_header(rom_bytes: bytes) -> RomHeader:
    title = rom_bytes[0x134:0x144].rstrip(b"\x00").decode("ascii", errors="replace")
    return RomHeader(
        title=title,
        cgb_flag=rom_bytes[0x143],
        sgb_flag=rom_bytes[0x146],
        cart_type=rom_bytes[0x147],
        rom_size=rom_bytes[0x148],
        ram_size=rom_bytes[0x149],
        destination=rom_bytes[0x14A],
        old_licensee=rom_bytes[0x14B],
        version=rom_bytes[0x14C],
        header_checksum=rom_bytes[0x14D],
        global_checksum=int.from_bytes(rom_bytes[0x14E:0x150], "big"),
        sha1=sha1(rom_bytes).hexdigest(),
    )


def verify_pokemon_red_rom(rom_bytes: bytes) -> RomVerification:
    header = read_rom_header(rom_bytes)
    matched_fields: list[str] = []
    warnings: list[str] = []

    expectations = {
        "title": "POKEMON RED",
        "cgb_flag": 0x00,
        "sgb_flag": 0x03,
        "cart_type": 0x13,
        "rom_size": 0x05,
        "ram_size": 0x03,
        "destination": 0x01,
        "version": 0x00,
        "global_checksum": 0x91E6,
    }
    for field, expected in expectations.items():
        actual = getattr(header, field)
        if actual == expected:
            matched_fields.append(field)
        else:
            warnings.append(f"{field}={actual!r} (expected {expected!r})")

    exact_sha1_match = header.sha1 == EXACT_ROM_SHA1
    if exact_sha1_match:
        matched_fields.append("sha1")
    else:
        warnings.append("sha1 did not match the known Pokemon Red (USA/Europe) revision 0 ROM dump")

    compatible = (
        header.title == expectations["title"]
        and header.version == expectations["version"]
        and header.destination == expectations["destination"]
    )
    return RomVerification(
        profile_id=PROFILE_ID,
        profile_name=PROFILE_NAME,
        compatible=compatible,
        exact_sha1_match=exact_sha1_match,
        matched_fields=tuple(matched_fields),
        warnings=tuple(warnings),
        header=header,
    )


def verify_pokemon_red_rom_file(path: str | Path) -> RomVerification:
    return verify_pokemon_red_rom(Path(path).read_bytes())


def rom_profile_metadata(verification: RomVerification | None) -> dict[str, object]:
    if verification is None:
        return {
            "id": PROFILE_ID,
            "name": PROFILE_NAME,
            "compatible": None,
            "exact_sha1_match": None,
        }
    return {
        "id": verification.profile_id,
        "name": verification.profile_name,
        "compatible": verification.compatible,
        "exact_sha1_match": verification.exact_sha1_match,
        "rom_title": verification.header.title,
        "rom_version": verification.header.version,
    }


def build_ram_context(memory) -> dict[str, object]:
    raw = {key: _read_field(memory, key) for key in FIELD_MAP}
    options = _decode_options(raw["options"])
    badges = _decode_badges(raw["obtained_badges"])
    money = _decode_bcd(raw["money_1"], raw["money_2"], raw["money_3"])
    battle_kind = _decode_battle_kind(raw["is_in_battle"])
    battle_context = _decode_battle_context(memory, battle_kind) if raw["is_in_battle"] != 0 else None
    party = _decode_party(memory, raw["party_count"])
    inventory = _decode_inventory(memory, raw["bag_count"])
    story = _decode_story_flags(memory)
    battle = {
        "flag": raw["is_in_battle"],
        "in_battle": raw["is_in_battle"] != 0,
        "kind": battle_kind,
    }
    if battle_context is not None and _battle_context_has_signal(battle_context):
        battle["context"] = battle_context

    return {
        "profile_id": PROFILE_ID,
        "player": {
            "map_id": raw["current_map"],
            "x": raw["player_x"],
            "y": raw["player_y"],
            "block_x": raw["player_block_x"],
            "block_y": raw["player_block_y"],
            "party_count": raw["party_count"],
            "bag_count": raw["bag_count"],
            "money": money,
            "badges": badges,
            "moving_direction_raw": raw["player_moving_direction"],
            "direction_raw": raw["player_direction"],
        },
        "party": party,
        "inventory": inventory,
        "map": {
            "tileset": raw["map_tileset"],
            "height": raw["map_height"],
            "width": raw["map_width"],
            "palette": raw["map_palette"],
        },
        "ui": {
            "window_y": raw["window_y"],
            "joy_ignore": raw["joy_ignore"],
            "input_locked": raw["joy_ignore"] != 0,
            "top_menu_item_y": raw["top_menu_item_y"],
            "top_menu_item_x": raw["top_menu_item_x"],
            "current_menu_item": raw["current_menu_item"],
            "max_menu_item": raw["max_menu_item"],
            "options": options,
        },
        "battle": battle,
        "story": story,
    }


def _read_field(memory, key: str) -> int:
    return int(memory[FIELD_MAP[key].address])


def _decode_bcd(*values: int) -> int | None:
    digits: list[str] = []
    for value in values:
        upper = (value >> 4) & 0x0F
        lower = value & 0x0F
        if upper > 9 or lower > 9:
            return None
        digits.extend((str(upper), str(lower)))
    return int("".join(digits))


def _decode_options(value: int) -> dict[str, object]:
    text_speed = value & 0x0F
    return {
        "raw": value,
        "battle_animation": "OFF" if value & 0x80 else "ON",
        "battle_style": "SET" if value & 0x40 else "SHIFT",
        "text_speed_raw": text_speed,
        "text_speed_preset": TEXT_SPEED_PRESETS.get(text_speed),
    }


def _decode_badges(value: int) -> dict[str, object]:
    names = [name for index, name in enumerate(BADGE_NAMES) if value & (1 << index)]
    return {
        "raw": value,
        "count": len(names),
        "names": names,
    }


def _decode_battle_kind(value: int) -> str | None:
    if value == 0:
        return None
    if value == 1:
        return "WILD"
    if value == 2:
        return "TRAINER"
    if value == 0xFF:
        return "LOST"
    return f"UNKNOWN_{value}"


def _decode_battle_context(memory, battle_kind: str | None) -> dict[str, object]:
    enemy_species_id = int(memory[ENEMY_SPECIES_ADDR])
    enemy_hp = _read_u16(memory, ENEMY_HP_ADDR)
    enemy_max_hp = _read_u16(memory, ENEMY_MAX_HP_ADDR)
    player_species_id = int(memory[PLAYER_ACTIVE_SPECIES_ADDR])
    player_hp = _read_u16(memory, PLAYER_ACTIVE_HP_ADDR)
    player_max_hp = _read_u16(memory, PLAYER_ACTIVE_MAX_HP_ADDR)
    battle_menu_position = int(memory[CURRENT_MENU_ITEM_ADDR])
    move_cursor_position = int(memory[REQUESTED_MOVE_CURSOR_ADDR])
    moves: list[dict[str, object]] = []

    for index in range(4):
        move_id = int(memory[PLAYER_ACTIVE_MOVES_ADDR + index])
        if move_id == 0:
            continue
        move_symbol, power, move_type = MOVE_DATA.get(move_id, (f"MOVE_{move_id:02X}", 0, "NORMAL"))
        pp = int(memory[PLAYER_ACTIVE_PP_ADDR + index]) & 0x3F
        moves.append(
            {
                "move_id": move_id,
                "name": _humanize_symbol(move_symbol),
                "pp": pp,
                "power": power,
                "move_type": move_type,
            }
        )

    enemy_species = None if enemy_species_id in {0, 0xFF} else _lookup_species_name(enemy_species_id)
    player_species = None if player_species_id in {0, 0xFF} else _lookup_species_name(player_species_id)

    return {
        "kind": battle_kind or "UNKNOWN",
        "opponent": enemy_species,
        "opponent_level": int(memory[ENEMY_LEVEL_ADDR]) or None,
        "moves": [move["name"] for move in moves],
        "enemy_species": enemy_species,
        "enemy_level": int(memory[ENEMY_LEVEL_ADDR]) or None,
        "enemy_hp": enemy_hp,
        "enemy_max_hp": enemy_max_hp,
        "player_active_species": player_species,
        "player_active_level": int(memory[PLAYER_ACTIVE_LEVEL_ADDR]) or None,
        "player_active_hp": player_hp,
        "player_active_max_hp": player_max_hp,
        "available_moves": moves,
        # Preserve the user-requested raw bytes while also exposing the verified cursor slot.
        "battle_menu_state": int(memory[PLAYER_ACTIVE_SPECIES_ADDR]),
        "battle_menu_position": battle_menu_position,
        "move_cursor_position": move_cursor_position,
        "battle_menu_saved_position": int(memory[BATTLE_SAVED_MENU_ITEM_ADDR]),
    }


def _battle_context_has_signal(context: dict[str, object]) -> bool:
    meaningful_keys = (
        "enemy_species",
        "enemy_level",
        "enemy_hp",
        "enemy_max_hp",
        "player_active_species",
        "player_active_level",
        "player_active_hp",
        "player_active_max_hp",
    )
    if any(context.get(key) not in {None, 0} for key in meaningful_keys):
        return True
    available_moves = context.get("available_moves")
    return isinstance(available_moves, list) and len(available_moves) > 0


def _decode_party(memory, party_count: int) -> list[dict[str, object]]:
    members: list[dict[str, object]] = []
    count = max(0, min(int(party_count), PARTY_LIMIT))
    for index in range(count):
        base = PARTY_MONS_ADDR + (index * PARTYMON_STRUCT_LENGTH)
        species_id = int(memory[base]) or int(memory[PARTY_SPECIES_ADDR + index])
        if species_id in {0, 0xFF}:
            continue
        hp = _read_u16(memory, base + MON_HP_OFFSET)
        max_hp = _read_u16(memory, base + MON_MAX_HP_OFFSET)
        status_raw = int(memory[base + MON_STATUS_OFFSET])
        members.append(
            {
                "name": _lookup_species_name(species_id),
                "species_id": species_id,
                "level": int(memory[base + MON_LEVEL_OFFSET]),
                "hp": hp,
                "max_hp": max_hp,
                "status": _decode_status(status_raw),
            }
        )
    return members


def _decode_inventory(memory, bag_count: int) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    count = max(0, min(int(bag_count), BAG_ITEM_CAPACITY))
    for index in range(count):
        base = BAG_ITEMS_ADDR + (index * 2)
        item_id = int(memory[base])
        if item_id in {0, 0xFF}:
            break
        items.append(
            {
                "name": _lookup_item_name(item_id),
                "item_id": item_id,
                "count": int(memory[base + 1]),
            }
        )
    return items


def _decode_story_flags(memory) -> dict[str, object]:
    watched_flags = {slug: _read_flag(memory, bit_index) for slug, (bit_index, _constant_name) in WATCHED_EVENT_FLAGS.items()}
    active_flags = [slug for slug, enabled in watched_flags.items() if enabled]
    return {
        "watched_flags": watched_flags,
        "active_flags": active_flags,
        "active_flag_labels": [_format_slug(slug) for slug in active_flags],
        "active_count": len(active_flags),
    }


def _lookup_species_name(species_id: int) -> str:
    if 0 <= species_id < len(POKEMON_SYMBOLS):
        symbol = POKEMON_SYMBOLS[species_id]
        if symbol:
            return _humanize_symbol(symbol)
    return f"SPECIES_{species_id:02X}"


def _lookup_item_name(item_id: int) -> str:
    if 0 <= item_id < len(ITEM_SYMBOLS):
        symbol = ITEM_SYMBOLS[item_id]
        if symbol:
            return _humanize_symbol(symbol)
    return f"ITEM_{item_id:02X}"


def _humanize_symbol(symbol: str) -> str:
    if symbol in SPECIAL_NAME_OVERRIDES:
        return SPECIAL_NAME_OVERRIDES[symbol]
    if symbol.startswith("TM_"):
        return f"TM {symbol[3:].replace('_', ' ').title()}"
    if symbol.startswith("HM_"):
        return f"HM {symbol[3:].replace('_', ' ').title()}"
    return symbol.replace("_", " ").title()


def _format_slug(slug: str) -> str:
    return slug.replace("_", " ").title()


def _decode_status(value: int) -> str | None:
    if value == 0:
        return None
    if value & 0x07:
        return "SLP"
    if value & 0x40:
        return "PAR"
    if value & 0x20:
        return "FRZ"
    if value & 0x10:
        return "BRN"
    if value & 0x08:
        return "PSN"
    return f"RAW_{value:02X}"


def _read_u16(memory, address: int) -> int:
    low = int(memory[address])
    high = int(memory[address + 1])
    return (high << 8) | low


def _read_flag(memory, bit_index: int) -> bool:
    address = EVENT_FLAGS_ADDR + (bit_index // 8)
    mask = 1 << (bit_index % 8)
    return bool(int(memory[address]) & mask)
