# 04-rom-data-import.md

## Objective

Import critical game data from the pokered disassembly (warp metadata, map objects, full type chart) to replace heuristic inference with authoritative data.

## Why this milestone exists

The agent currently guesses connector activation modes (push vs step_on vs interact) based on heuristics like boundary position and blocked neighbor count. This causes stuck loops when the guess is wrong. The ROM data has explicit warp definitions, door tile IDs, and NPC/sign positions that make these decisions data-driven.

Similarly, the battle type chart is incomplete in the current `battle_manager.py`. The full Gen 1 type chart is available in the disassembly.


## Github repo
Link: https://github.com/pret/pokered/tree/master/data

Ensure the import will import the github repo and then remove it afterwards.

## Scope

- Import warp definitions per map (destination, tile position, activation)
- Import NPC and sign positions per map
- Import door tile IDs and warp tile IDs per tileset
- Import full Gen 1 type effectiveness chart
- Create `GameKnowledge` service that indexes imported data
- Replace heuristic connector activation with data-driven lookup

## Non-goals

- Importing trainer parties (helpful later, not essential now)
- Importing wild encounter tables (helpful later)
- Importing item data beyond what's already used
- Importing text banks or dialogue scripts
- Importing Pokemon species data (learnsets, base stats)

## Current code to inspect first

- `docs/reference/pokered_data_inventory.md`: data inventory (the source for all recommendations below)
- `scripts/import_pret_world_graph.py`: existing import script
- `src/pokemon_agent/generated/world_graph.json`: current generated data
- `src/pokemon_agent/agent/world_map.py`: `_detect_connectors()`, `_transition_connector_spec()`, `_canonical_connector_hints()`
- `src/pokemon_agent/agent/battle_manager.py`: `TYPE_EFFECTIVENESS` dict

## Concrete code changes

### 1. Extend import script

Modify `scripts/import_pret_world_graph.py` to additionally parse:

**Map objects** (from `data/maps/objects/*.asm`):
Each map's object file contains:
- Warp definitions: destination map, destination warp index, tile coordinates
- Sign definitions: tile coordinates (signs are interactable tiles)
- NPC definitions: sprite index, position, movement pattern, text script
- Object events: item pickups, hidden items

Output: `src/pokemon_agent/generated/map_objects.json`
```json
{
  "PalletTown": {
    "warps": [
      {"x": 5, "y": 3, "dest_map": "RedsHouse1F", "dest_warp_index": 0}
    ],
    "signs": [
      {"x": 3, "y": 5, "text_id": 1}
    ],
    "npcs": [
      {"sprite_id": 1, "x": 8, "y": 5, "text_id": 1, "trainer": false}
    ]
  }
}
```

**Type chart** (from `data/types/type_matchups.asm`):
Format: attacker_type, defender_type, effectiveness (0/5/10/20 → 0x/0.5x/1x/2x)

Output: `src/pokemon_agent/generated/type_chart.json`
```json
{
  "matchups": [
    {"attacker": "FIRE", "defender": "GRASS", "multiplier": 2.0},
    {"attacker": "WATER", "defender": "FIRE", "multiplier": 2.0},
    ...
  ]
}
```

**Warp tile IDs** (from `data/tilesets/warp_tile_ids.asm`, `door_tile_ids.asm`):
Which collision tile IDs are warp triggers or doors per tileset.

Output: include in `map_objects.json` or separate `tilesets.json`

### 2. GameKnowledge service (`agent/game_knowledge.py`)

```python
class GameKnowledge:
    def __init__(self):
        self._world_graph = load_world_graph()
        self._map_objects = self._load_map_objects()
        self._type_chart = self._load_type_chart()
        # Index for fast lookup
        self._warps_by_map: dict[str, list[WarpDef]] = ...
        self._npcs_by_map: dict[str, list[NPCDef]] = ...
        self._signs_by_map: dict[str, list[SignDef]] = ...
        self._effectiveness: dict[tuple[str,str], float] = ...

    def get_warp_at(self, map_name, x, y) -> WarpDef | None: ...
    def get_warps_on_map(self, map_name) -> list[WarpDef]: ...
    def get_npcs_on_map(self, map_name) -> list[NPCDef]: ...
    def get_signs_on_map(self, map_name) -> list[SignDef]: ...
    def type_effectiveness(self, attack_type, defend_type) -> float: ...
    def is_super_effective(self, attack_type, defend_type) -> bool: ...
```

### 3. Replace heuristic connector activation

In `world_map.py`, replace `_detect_connectors()` heuristic with data lookup:
```python
# Before: guess activation mode from boundary position
# After: look up warp metadata from GameKnowledge
warp = game_knowledge.get_warp_at(map_name, x, y)
if warp is not None:
    activation_mode = "step_on"  # warps are step-on by default
    dest_map = warp.dest_map
```

### 4. Wire type chart into BattleController

Replace hardcoded `TYPE_EFFECTIVENESS` dict with `GameKnowledge.type_effectiveness()`.

## Interfaces and contracts

### GameKnowledge
- Singleton, loaded once at startup
- All lookups return None or empty list if data missing (not error)
- Map names use the same normalization as world_graph.py
- Type names use Gen 1 names (NORMAL, FIRE, WATER, etc.)

### WarpDef, NPCDef, SignDef
Simple frozen dataclasses:
```python
@dataclass(frozen=True)
class WarpDef:
    x: int
    y: int
    dest_map: str
    dest_warp_index: int

@dataclass(frozen=True)
class NPCDef:
    sprite_id: int
    x: int
    y: int
    is_trainer: bool
    text_id: int | None = None

@dataclass(frozen=True)
class SignDef:
    x: int
    y: int
    text_id: int | None = None
```

## Data model changes

### New models
- `WarpDef`, `NPCDef`, `SignDef` (simple data containers)
- `GameKnowledge` service class

### Modified
- `DiscoveredConnector.activation_mode`: now sourced from ROM data when available

## Migration plan

1. Write/extend import script (can run independently)
2. Generate JSON files (verify against manual spot-checks)
3. Create GameKnowledge service with tests
4. Wire into world_map.py (replace heuristic activation)
5. Wire into BattleController (replace hardcoded type chart)
6. Verify no regressions

## Tests and evaluation

### Import script tests
- Generated JSON parses without errors
- Spot-check 10 maps: warp count matches expected
- Type chart: 15 types × 15 types = 225 entries (minus self-neutral)
- Known warps verified: PalletTown has exits to Route 1, Red's House, Blue's House

### GameKnowledge tests
- `get_warp_at("PalletTown", x, y)` returns correct warp for known positions
- `type_effectiveness("FIRE", "GRASS")` returns 2.0
- `type_effectiveness("NORMAL", "GHOST")` returns 0.0
- Missing map returns empty list
- Missing type pair returns 1.0 (neutral)

### Integration tests
- Connector activation uses warp data instead of heuristic
- Battle move scoring uses full type chart

## Risks and edge cases

- **Parsing assembly is non-trivial.** The import script must handle macros, includes, and symbolic constants. Mitigate: the existing script already does this for world_graph; extend the same approach.
- **Map name mismatches between ROM data and runtime.** ROM uses symbols like "PALLET_TOWN", runtime uses "Pallet Town". Mitigate: normalize during import using existing fuzzy matching.
- **NPC positions may differ from runtime.** ROM has initial positions; NPCs can move during gameplay. Mitigate: use ROM positions as defaults, validate against runtime sprite data.
- **Type chart edge cases.** Gen 1 has unique mechanics (Psychic immune to Ghost due to bug, Critical hits ignore stat stages). Mitigate: import raw chart, document known quirks.

## Acceptance criteria

1. `map_objects.json` generated with warp/NPC/sign data for all maps
2. `type_chart.json` generated with all Gen 1 matchups
3. `GameKnowledge` provides O(1) lookup by (map, x, y) for warps
4. `GameKnowledge` provides type effectiveness lookup
5. Connector activation modes sourced from data (not heuristic) when data available
6. Battle type chart is complete (all 15 Gen 1 types)
7. All tests pass

## Rollback / fallback notes

- GameKnowledge lookups fall back to heuristic when data is missing
- Old connector detection logic preserved as fallback path
- Type chart can fall back to hardcoded dict if import fails

## Ordered implementation checklist

1. **Step 4.1: Parse map object ASM files**
   - Action: Extend import script to parse `data/maps/objects/*.asm`
   - Files: `scripts/import_pret_world_graph.py`
   - Dependencies: none
   - Done: warps, NPCs, signs extracted from ASM
   - Tests: spot-check parse output for 10 maps

2. **Step 4.2: Generate map_objects.json**
   - Action: Write parsed data to structured JSON
   - Files: `generated/map_objects.json`
   - Dependencies: 4.1
   - Done: JSON file exists, valid schema
   - Tests: JSON schema validation

3. **Step 4.3: Parse type chart**
   - Action: Parse `data/types/type_matchups.asm`
   - Files: `scripts/import_pret_world_graph.py`, `generated/type_chart.json`
   - Dependencies: none
   - Done: all matchups extracted
   - Tests: count matchups, verify known pairs

4. **Step 4.4: Create GameKnowledge service**
   - Action: Load and index all generated JSON
   - Files: `agent/game_knowledge.py`
   - Dependencies: 4.2, 4.3
   - Done: all lookups work
   - Tests: `test_game_knowledge.py`

5. **Step 4.5: Replace heuristic connector activation**
   - Action: Use GameKnowledge warp data in world_map.py
   - Files: `agent/world_map.py`
   - Dependencies: 4.4
   - Done: activation modes data-driven
   - Tests: connector activation tests

6. **Step 4.6: Wire type chart into BattleController**
   - Action: Replace hardcoded TYPE_EFFECTIVENESS
   - Files: `controllers/battle.py`
   - Dependencies: 4.4, Phase 2
   - Done: full chart coverage
   - Tests: type matchup edge cases
