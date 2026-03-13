# Milestone 02 — Nail pulling game state

## Goal
Prove that the engine can extract stable, structured game state from the emulator.

## Outcomes
- structured state object exists
- initial extraction works for overworld basics
- step counter and metadata are recorded
- state snapshots can be printed or serialized
- tile-map and/or RAM hooks are explored and documented

## Required work
Implement or stub the path for extracting:
- map identifier or best available placeholder
- player x/y if available
- facing direction if available
- high-level mode (`OVERWORLD`, `MENU`, `TEXT`, `BATTLE`, `UNKNOWN`)
- menu/text flags if available

## Recommended approach
- prefer machine-readable state over screenshots
- if full Pokémon RAM mapping is not ready, build the extraction pipeline first and mark unresolved fields explicitly
- document exactly what is real and what is unknown

## Definition of done
- `get_structured_state()` returns a typed object
- repeated reads across multiple steps work without crashing
- code clearly separates confirmed fields vs placeholders

## Stretch
- add one debug script to dump state every N frames
