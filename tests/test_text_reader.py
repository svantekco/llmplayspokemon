from __future__ import annotations

import numpy as np

from pokemon_agent.emulator.text_reader import decode_screen_text
from pokemon_agent.emulator.text_reader import detect_yes_no_prompt


_ENCODE = {
    " ": 0x7F,
    "'": 0xE0,
    "!": 0xE7,
    "?": 0xE6,
}
_ENCODE.update({letter: 0x80 + index for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})
_ENCODE.update({letter: 0xA0 + index for index, letter in enumerate("abcdefghijklmnopqrstuvwxyz")})


def _text_box(lines: list[str], *, yes_no_prompt: bool = False) -> np.ndarray:
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
            grid[row_index, col_index] = _ENCODE[char]

    if yes_no_prompt:
        grid[14, 12] = 0xED
        for offset, char in enumerate("YES", start=13):
            grid[14, offset] = _ENCODE[char]
        for offset, char in enumerate("NO", start=13):
            grid[15, offset] = _ENCODE[char]
    return grid


def test_decode_screen_text_reads_bottom_text_box() -> None:
    grid = _text_box(["Hello!", "Oak's Parcel"])

    text = decode_screen_text(grid)

    assert text == "Hello!\nOak's Parcel"


def test_detect_yes_no_prompt_requires_arrow_and_words() -> None:
    grid = _text_box(["Buy item?"], yes_no_prompt=True)

    assert detect_yes_no_prompt(grid) is True


def test_decode_screen_text_returns_none_without_text_box() -> None:
    grid = np.zeros((18, 20), dtype=np.uint32)

    assert decode_screen_text(grid) is None
    assert detect_yes_no_prompt(grid) is False
