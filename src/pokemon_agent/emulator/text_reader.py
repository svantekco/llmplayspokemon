from __future__ import annotations

import re
from typing import Iterable

TEXT_BOX_ROWS = 6
TEXT_BOX_BORDER_TILES = {0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E}
YES_NO_ARROW_TILES = {0xEC, 0xED, 0xEE}

POKEMON_RED_CHAR_MAP: dict[int, str] = {
    0x70: "to",
    0x71: "'",
    0x72: '"',
    0x73: '"',
    0x75: "...",
    0x79: " ",
    0x7A: " ",
    0x7B: " ",
    0x7C: " ",
    0x7D: " ",
    0x7E: " ",
    0x7F: " ",
    0x9A: "(",
    0x9B: ")",
    0x9C: ":",
    0x9D: ";",
    0x9E: "[",
    0x9F: "]",
    0xBA: "e",
    0xBB: "'d",
    0xBC: "'l",
    0xBD: "'s",
    0xBE: "'t",
    0xBF: "'v",
    0xE0: "'",
    0xE3: "-",
    0xE4: "'r",
    0xE5: "'m",
    0xE6: "?",
    0xE7: "!",
    0xE8: ".",
    0xEC: " ",
    0xED: " ",
    0xEE: " ",
    0xEF: "m",
    0xF0: "Y",
    0xF1: "x",
    0xF2: ".",
    0xF3: "/",
    0xF4: ",",
    0xF5: "f",
}

for offset, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    POKEMON_RED_CHAR_MAP[0x80 + offset] = letter

for offset, letter in enumerate("abcdefghijklmnopqrstuvwxyz"):
    POKEMON_RED_CHAR_MAP[0xA0 + offset] = letter

for offset, digit in enumerate("0123456789"):
    POKEMON_RED_CHAR_MAP[0xF6 + offset] = digit


def decode_screen_text(game_area_tiles: Iterable[Iterable[int]]) -> str | None:
    text_rows = _bottom_text_rows(game_area_tiles)
    if not _looks_like_text_box(text_rows):
        return None

    decoded_lines: list[str] = []
    for row in text_rows:
        decoded = "".join(POKEMON_RED_CHAR_MAP.get(tile, " ") for tile in row)
        normalized = _normalize_line(decoded)
        if normalized:
            decoded_lines.append(normalized)

    if not decoded_lines:
        return None

    text = "\n".join(decoded_lines)
    return text if re.search(r"[A-Za-z0-9]", text) else None


def detect_yes_no_prompt(game_area_tiles: Iterable[Iterable[int]]) -> bool:
    text_rows = _bottom_text_rows(game_area_tiles)
    if not _looks_like_text_box(text_rows):
        return False

    has_arrow = any(tile in YES_NO_ARROW_TILES for row in text_rows for tile in row)
    if not has_arrow:
        return False

    decoded_lines = []
    for row in text_rows:
        decoded = "".join(POKEMON_RED_CHAR_MAP.get(tile, " ") for tile in row)
        normalized = _normalize_line(decoded).upper()
        if normalized:
            decoded_lines.append(normalized)

    if not decoded_lines:
        return False

    flattened = " ".join(decoded_lines)
    return "YES" in flattened and re.search(r"\bNO\b", flattened) is not None


def _bottom_text_rows(game_area_tiles: Iterable[Iterable[int]]) -> list[list[int]]:
    rows = [[int(tile) for tile in row] for row in game_area_tiles]
    if not rows:
        return []
    return rows[-TEXT_BOX_ROWS:]


def _looks_like_text_box(text_rows: list[list[int]]) -> bool:
    if not text_rows:
        return False
    border_tiles = sum(tile in TEXT_BOX_BORDER_TILES for row in text_rows for tile in row)
    text_tiles = sum(tile in POKEMON_RED_CHAR_MAP and tile not in TEXT_BOX_BORDER_TILES for row in text_rows for tile in row)
    return border_tiles >= 8 and text_tiles >= 3


def _normalize_line(value: str) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    compact = compact.replace(" '", "'")
    compact = compact.replace('" ', '"')
    return compact
