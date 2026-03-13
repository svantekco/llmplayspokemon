# Milestone 01 — Install and connect the Game Boy emulator

## Goal
Install and connect **PyBoy** as the initial emulator target and prove the program can boot a ROM and step frames.

## Outcomes
- PyBoy installed and importable
- local ROM path accepted from CLI
- emulator process boots successfully
- can advance frames deterministically
- can send at least simple button inputs
- can optionally save and load state

## Required work
- implement `PyBoyAdapter` bootstrap path
- expose methods:
  - `boot(rom_path)`
  - `advance_frames(n)`
  - `press_button(button)`
  - `get_raw_state()`
- log emulator startup and failure reasons clearly

## Guardrails
- do not attempt LLM integration here
- do not rely on screenshots for proving emulator access
- keep this milestone focused on deterministic control

## Definition of done
- program boots Pokémon Red ROM through PyBoy
- can advance frames and press buttons from code
- logs confirm action execution
- no GUI interaction required once started

## Stretch
- add a script that performs a deterministic input demo sequence
