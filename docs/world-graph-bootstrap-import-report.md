# World Graph Bootstrap Import Report

## Summary

This change adds a project-owned static world graph and landmark registry derived from the canonical PRET `pokered` disassembly as a one-time bootstrap import.

The runtime does not parse PRET ASM files and does not require a checked-in PRET checkout. The repo now carries generated JSON artifacts plus a regeneration script.

## Existing code/data found

Before the import, the repo already had:

- Walkthrough and progression logic in `src/pokemon_agent/data/walkthrough.py`
- Existing map name data in `src/pokemon_agent/data/map_names.py`
- A hand-maintained route table in `src/pokemon_agent/data/map_connections.py`
- Dynamic discovered-world memory in `src/pokemon_agent/agent/world_map.py`
- Local tile pathfinding in `src/pokemon_agent/agent/navigation.py`
- Planner/runtime integration in `src/pokemon_agent/agent/engine.py`
- LLM prompt/context assembly in `src/pokemon_agent/agent/context_manager.py`

## What was reused

The implementation deliberately reused the project’s existing structure instead of replacing it:

- `map_names.py` remains the display-name source for canonical map IDs already present in the repo.
- `world_map.py` remains the dynamic runtime memory of discovered connectors and visited local structure.
- `engine.py` still owns high-level navigation goal selection and local candidate ranking.
- `context_manager.py` still owns prompt/context shaping.
- `data/map_connections.py` remains the compatibility layer used by existing engine code, but now reads from the generated world graph instead of a hand-maintained table.

## What was generated from PRET

Source repository used:

- Repo: `https://github.com/pret/pokered`
- Imported from a temporary checkout during development
- Recorded source commit is embedded in the generated artifact metadata

Generated artifacts:

- `src/pokemon_agent/generated/world_graph.json`
- `src/pokemon_agent/generated/landmarks.json`

Imported data includes:

- Canonical map IDs, symbols, names, sizes, and tilesets
- Directional map connections from PRET map headers
- Warp events with coordinates and resolved destinations
- Landmark registry inferred from canonical destination map names, tilesets, warp targets, and sign/background events

Current generated stats:

- 248 maps
- 78 directional connections
- 802 resolved warps
- 3 unresolved special-case `LAST_MAP` warps
- 451 landmarks

The remaining unresolved warps are special/unused cases that do not block the new early-game navigation features. They are left explicit with `null` destinations in the generated data instead of being guessed silently.

## Modules added

New code added:

- `scripts/import_pret_world_graph.py`
- `src/pokemon_agent/navigation/world_graph.py`
- `src/pokemon_agent/navigation/__init__.py`
- `src/pokemon_agent/generated/__init__.py`

Key responsibilities:

- `scripts/import_pret_world_graph.py`
  - Reads PRET map constants, map headers, object/warp/bg-event files
  - Resolves most `LAST_MAP` warps from canonical warp tables
  - Emits project-local JSON artifacts
- `navigation/world_graph.py`
  - Loads generated artifacts
  - Exposes deterministic query APIs for maps, warps, landmarks, nearest-service lookup, and route planning
- `data/map_connections.py`
  - Preserves the existing route-table API surface while delegating to the new world graph

## Planner integration

Planner behavior remains additive and minimal:

- High-level objective selection still comes from the existing walkthrough/progression logic.
- The engine now uses the generated static world graph as its canonical map/warp reference.
- Canonical warp entrances are available as connector candidates even before the runtime has dynamically discovered them.
- Story milestones on city maps can now resolve to concrete target landmarks when the milestone text clearly implies one.

Examples:

- Brock progression on `Pewter City` now resolves toward the canonical `pewter_city_gym` landmark.
- Parcel delivery on `Viridian City` can resolve toward the canonical mart landmark.
- Nearest Pokecenter lookup is available from the static landmark registry.

Dynamic and static state remain separate:

- Static: generated world graph, warp topology, landmark registry
- Dynamic: current map, current position, discovered connectors, flags, items, progress, and current navigation goal

## LLM context integration

The prompt builder still produces a compact context payload.

It now adds a concise `canonical_navigation` block under `overworld_context` with:

- `current_map`
- `target_map`
- `target_landmark`
- `route_summary`
- `nearest_pokecenter`
- `current_map_neighbors`

This provides grounded navigation context without dumping the full world graph into the prompt.

## Tests added

Added coverage in `tests/test_world_graph.py` for:

- Early-game routing
- Warp resolution
- Viridian City landmark classification
- Nearest Pokecenter lookup
- Planner integration via landmark-targeted navigation goals
- Prompt/context navigation grounding

## Regeneration steps

The default checked-in repo state does not require PRET.

To regenerate later:

1. Clone PRET `pokered` somewhere outside this repo.
2. Run:

   ```bash
   python3 scripts/import_pret_world_graph.py --pret-dir /path/to/pokered
   ```

3. Review the updated generated JSON artifacts.
4. Run the relevant tests.

No PRET files need to be copied into this repo.

## Temporary dependency removal

The PRET checkout used for the bootstrap import was temporary and is not part of the final repo state.

No PRET submodule, vendor copy, or runtime ASM dependency was left behind.
