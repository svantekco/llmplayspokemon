# pokered `data/` inventory

This note is meant to help another coding model understand what is available under `pret/pokered/data` and what is worth importing into structured tables.

## What this subtree contains

`data/` is the main gameplay/content data area for the disassembly. It is mostly assembly-source data (`.asm`) organized by domain:

- battle logic tables
- battle animation tables
- credits data
- scripted and hidden event data
- item catalogs, pricing, and shop inventories
- map metadata and per-map object/header data
- move definitions and move-related lookup tables
- player-name text data
- Pokemon species data
- Super Game Boy presentation data
- sprite-facing and sprite metadata
- text banks and text-related lookup tables
- tileset/collision/warp metadata
- trainer classes, parties, AI, and related tables
- type chart data
- wild encounter data
- a few root-level UI/system pointer tables

## Best import targets

If the goal is to import gameplay content into a database or JSON bundle, these are the highest-value datasets:

1. **Pokemon species data**
2. **Move data**
3. **Item data and marts**
4. **Trainer classes and trainer parties**
5. **Map list + map headers + map objects**
6. **Wild encounter tables**
7. **Type chart**
8. **Hidden items / hidden events / NPC trades**
9. **UI text and menu strings**
10. **Growth-rate formulas**

## Import-friendly breakdown by path

### `data/pokemon/`

This is the richest gameplay dataset.

#### Main files

- `data/pokemon/base_stats.asm`
  - species include/pointer list for the per-species base stat files
- `data/pokemon/evos_moves.asm`
  - evolution rules and learnsets
- `data/pokemon/names.asm`
  - species names
- `data/pokemon/dex_entries.asm`
  - pointer/include structure for Pokedex entry content
- `data/pokemon/dex_text.asm`
  - Pokedex text strings
- `data/pokemon/dex_order.asm`
  - Pokedex ordering
- `data/pokemon/cries.asm`
  - cry assignments / cry metadata
- `data/pokemon/menu_icons.asm`
  - per-species menu icon mapping
- `data/pokemon/palettes.asm`
  - palette data/mapping
- `data/pokemon/title_mons.asm`
  - title-screen Pokemon selection/order
- `data/pokemon/mew.asm`
  - special handling/data for Mew

#### `data/pokemon/base_stats/`

One file per species, for example `bulbasaur.asm`, `abra.asm`, `articuno.asm`, etc.

A per-species base-stats file is highly structured and typically contains:

- Pokedex id
- HP / Attack / Defense / Speed / Special
- primary and secondary type
- catch rate
- base EXP
- sprite dimension byte / sprite pointers
- level-1 learnset
- growth rate
- TM/HM compatibility bitset/list

**Suggested tables**

- `pokemon_species`
- `pokemon_base_stats`
- `pokemon_types`
- `pokemon_growth_rates`
- `pokemon_tmhm_compatibility`
- `pokemon_level1_moves`
- `pokemon_evolutions`
- `pokemon_levelup_moves`
- `pokedex_entries`

---

### `data/moves/`

Move definitions and move-related metadata.

#### Files

- `data/moves/moves.asm`
  - core move table: animation id, effect, power, type, accuracy, PP
- `data/moves/names.asm`
  - move names
- `data/moves/effects_pointers.asm`
  - pointers to effect handlers
- `data/moves/animations.asm`
  - move animation lookup/mapping
- `data/moves/field_move_names.asm`
  - names for field moves
- `data/moves/field_moves.asm`
  - field move behavior/mapping
- `data/moves/grammar.asm`
  - text grammar helpers for move names/messages
- `data/moves/hm_moves.asm`
  - HM move list
- `data/moves/tmhm_moves.asm`
  - TM/HM move mapping/list
- `data/moves/sfx.asm`
  - per-move or animation sound-effect metadata

**Suggested tables**

- `moves`
- `move_names`
- `move_effects`
- `field_moves`
- `tmhm_moves`

---

### `data/items/`

Item catalog, prices, and shop data.

#### Files

- `data/items/names.asm`
  - item names
- `data/items/prices.asm`
  - base item prices
- `data/items/tm_prices.asm`
  - TM prices
- `data/items/vending_prices.asm`
  - vending-machine prices
- `data/items/marts.asm`
  - mart inventories by clerk/location
- `data/items/key_items.asm`
  - key-item list / grouping
- `data/items/guard_drink_items.asm`
  - the drinks that satisfy gate guards
- `data/items/use_overworld.asm`
  - overworld-use behavior
- `data/items/use_party.asm`
  - in-party-use behavior

**Suggested tables**

- `items`
- `item_prices`
- `mart_inventories`
- `item_use_overworld`
- `item_use_party`
- `key_items`

---

### `data/trainers/`

Trainer classes, trainer AI, parties, and presentation data.

#### Files

- `data/trainers/names.asm`
  - trainer-class names
- `data/trainers/name_pointers.asm`
  - trainer name pointer table
- `data/trainers/parties.asm`
  - trainer parties; supports both fixed-level teams and explicit level/species pairs
- `data/trainers/ai_pointers.asm`
  - trainer AI pointer table
- `data/trainers/move_choices.asm`
  - AI move choice helpers/weights/rules
- `data/trainers/encounter_types.asm`
  - encounter typing/classification data
- `data/trainers/pic_pointers_money.asm`
  - trainer portrait pointers and prize money data
- `data/trainers/special_moves.asm`
  - special trainer move logic or move allowances

**Suggested tables**

- `trainer_classes`
- `trainer_names`
- `trainer_parties`
- `trainer_party_members`
- `trainer_ai`
- `trainer_rewards`

---

### `data/maps/`

Map metadata is split across pointer tables plus per-map files.

#### Root files

- `data/maps/names.asm`
  - display names for towns, routes, dungeons, ships, etc.
- `data/maps/map_header_pointers.asm`
  - master pointer table for map headers
- `data/maps/map_header_banks.asm`
  - bank table for map headers
- `data/maps/sprite_sets.asm`
  - sprite-set assignments per map/group
- `data/maps/songs.asm`
  - music assignments
- `data/maps/town_map_entries.asm`
  - town-map placement data
- `data/maps/town_map_order.asm`
  - ordering for town-map display/navigation
- `data/maps/special_warps.asm`
  - dungeon warp specs, fly destinations, new-game warp, cable club warps
- `data/maps/toggleable_objects.asm`
  - objects gated by events/state
- `data/maps/force_bike_surf.asm`
  - maps/areas that force bike or surf state
- `data/maps/badge_maps.asm`
  - badge-related map lists
- `data/maps/dungeon_maps.asm`
  - dungeon map lists
- `data/maps/rest_house_maps.asm`
  - rest house map lists

#### Subdirectories

- `data/maps/headers/`
  - appears to hold per-map header definitions referenced by `map_header_pointers.asm`
- `data/maps/objects/`
  - one file per map with warp/sign/NPC/object placement and related map object scripting data

**Suggested tables**

- `maps`
- `map_headers`
- `map_warps`
- `map_connections`
- `map_objects`
- `map_npcs`
- `map_signs`
- `map_music`
- `map_sprite_sets`
- `fly_warps`

---

### `data/wild/`

Wild encounter data, split into global tables and per-map files.

#### Files

- `data/wild/grass_water.asm`
  - global pointer table for wild data by map; format for grass and surfing encounters
- `data/wild/probabilities.asm`
  - encounter slot probability tables
- `data/wild/good_rod.asm`
  - Good Rod encounter data
- `data/wild/super_rod.asm`
  - Super Rod encounter data

#### `data/wild/maps/`

Per-map encounter files such as:

- `Route1.asm`
- `ViridianForest.asm`
- `MtMoon1F.asm`
- `PokemonTower1F.asm`
- `PowerPlant.asm`
- `CeruleanCave1F.asm`
- `SafariZoneCenter.asm`
- etc.

These files are the best place to import encounter slots by map.

**Suggested tables**

- `wild_encounter_maps`
- `wild_encounter_slots`
- `rod_encounters`
- `wild_encounter_probabilities`

---

### `data/events/`

Script-adjacent content tables and hidden-world data.

#### Files

- `data/events/hidden_events.asm`
  - hidden events by map; coordinates + event function references
- `data/events/hidden_item_coords.asm`
  - hidden item coordinate data
- `data/events/hidden_coins.asm`
  - hidden coin locations
- `data/events/card_key_coords.asm`
  - card-key door coordinates
- `data/events/card_key_maps.asm`
  - maps relevant to card-key handling
- `data/events/trades.asm`
  - in-game NPC trades (give species, receive species, dialog set, nickname)
- `data/events/prizes.asm`
  - prize-corner prizes
- `data/events/prize_mon_levels.asm`
  - prize Pokemon levels
- `data/events/slot_machine_wheels.asm`
  - slot-machine reel/wheel data
- `data/events/bench_guys.asm`
  - bench-guy text/event lookup data

**Suggested tables**

- `npc_trades`
- `hidden_items`
- `hidden_coins`
- `hidden_events`
- `game_corner_prizes`
- `slot_machine_tables`

---

### `data/types/`

Type names and type matchup matrix.

#### Files

- `data/types/names.asm`
  - type names
- `data/types/type_matchups.asm`
  - attacker/defender/effect triples for the Gen 1 type chart

**Suggested tables**

- `types`
- `type_matchups`

---

### `data/battle/`

Battle-effect classification and stat metadata.

#### Files

- `data/battle/always_happen_effects.asm`
- `data/battle/critical_hit_moves.asm`
- `data/battle/unused_critical_hit_moves.asm`
- `data/battle/residual_effects_1.asm`
- `data/battle/residual_effects_2.asm`
- `data/battle/set_damage_effects.asm`
- `data/battle/special_effects.asm`
- `data/battle/stat_names.asm`
- `data/battle/stat_mod_names.asm`
- `data/battle/stat_modifiers.asm`

These are mostly lookup/classification tables rather than player-facing content tables.

**Suggested tables**

- `battle_effect_categories`
- `critical_hit_move_flags`
- `stat_names`
- `stat_stage_modifiers`

---

### `data/battle_anims/`

Battle animation metadata.

#### Files

- `data/battle_anims/base_coords.asm`
- `data/battle_anims/frame_blocks.asm`
- `data/battle_anims/special_effect_pointers.asm`
- `data/battle_anims/special_effects.asm`
- `data/battle_anims/subanimations.asm`

Useful if you want to import animation structures, but lower priority for game-balance/content analysis.

---

### `data/text/`

Main text banks and text-system helpers.

#### Files

- `data/text/text_1.asm` through `data/text/text_7.asm`
  - large banks of game dialogue/text
- `data/text/alphabets.asm`
  - alphabet/charset data
- `data/text/dakutens.asm`
  - kana modifier data for JP text support
- `data/text/unused_names.asm`
  - unused names/text remnants

**Suggested tables**

- `text_entries`
- `text_banks`
- `char_map`

---

### `data/player/`

Player naming data.

#### Files

- `data/player/names.asm`
- `data/player/names_list.asm`

Good candidate for default/player/rival naming imports.

---

### `data/sprites/`

Sprite metadata.

#### Files

- `data/sprites/facings.asm`
  - facing tables / directional sprite-layout data
- `data/sprites/sprites.asm`
  - sprite metadata / sprite references

---

### `data/tilesets/`

Movement, collision, and environmental tile behavior.

#### Files

- `data/tilesets/tileset_headers.asm`
- `data/tilesets/collision_tile_ids.asm`
- `data/tilesets/pair_collision_tile_ids.asm`
- `data/tilesets/door_tile_ids.asm`
- `data/tilesets/warp_tile_ids.asm`
- `data/tilesets/warp_pad_hole_tile_ids.asm`
- `data/tilesets/warp_carpet_tile_ids.asm`
- `data/tilesets/bookshelf_tile_ids.asm`
- `data/tilesets/ledge_tiles.asm`
- `data/tilesets/spinner_tiles.asm`
- `data/tilesets/cut_tree_blocks.asm`
- `data/tilesets/water_tilesets.asm`
- `data/tilesets/dungeon_tilesets.asm`
- `data/tilesets/bike_riding_tilesets.asm`
- `data/tilesets/escape_rope_tilesets.asm`

These are ideal for importing environmental behavior rules.

---

### `data/sgb/`

Super Game Boy presentation data.

#### Files

- `data/sgb/sgb_border.asm`
- `data/sgb/sgb_packets.asm`
- `data/sgb/sgb_palettes.asm`

---

### `data/credits/`

Credits sequencing and strings.

#### Files

- `data/credits/credits_order.asm`
  - credit-sequence ordering
- `data/credits/credits_text.asm`
  - staff-role and staff-name strings
- `data/credits/credits_mons.asm`
  - Pokemon shown during credits

---

## Root-level files directly under `data/`

These are cross-cutting tables rather than domain-specific content folders.

- `data/growth_rates.asm`
  - XP growth-rate formulas
- `data/icon_pointers.asm`
  - party/menu icon graphics pointers and icon class metadata
- `data/predef_pointers.asm`
  - pointer table for predefined ASM routines used by scripts/engine systems
- `data/text_boxes.asm`
  - text-box layouts, menu coordinates, and embedded small menu strings
- `data/text_predef_pointers.asm`
  - predefined text handler pointer table
- `data/tilemaps.asm`
  - tilemap pointer table and included tilemap assets
- `data/yes_no_menu_strings.asm`
  - two-option menu string sets like YES/NO, TRADE/CANCEL, etc.

## Parsing notes for Codex / Claude

1. **Many files are declarative tables, not logic.**
   - `db`, `dw`, `dba`, macros, and pointer tables are common.

2. **A lot of content is normalized across multiple files.**
   - Example: maps are split into name tables, pointer tables, bank tables, header files, and object files.

3. **`INCLUDE` and per-entity files matter.**
   - Species data and wild encounter data are often assembled from include lists and per-entity/per-map files.

4. **Enums are symbolic, not numeric.**
   - Species, items, moves, maps, and types usually appear as symbolic constants. A parser should preserve both raw symbol and resolved numeric id when possible.

5. **Some tables are pointer/index tables rather than final records.**
   - For importing, it is usually better to materialize them into explicit records with foreign keys.

6. **Not every file is equally useful for a product-facing data import.**
   - The most useful imports are Pokemon, moves, items, trainers, maps, wild encounters, events, and types.
   - Lower-priority imports are battle animation internals, SGB data, and low-level pointer tables.

## Recommended first-pass import plan

### Tier 1

- Pokemon species + base stats
- evolutions + learnsets
- moves
- items + prices + marts
- trainer classes + parties
- map names + map headers + map objects
- wild encounters
- type chart

### Tier 2

- hidden items / hidden events / NPC trades / prize corner
- text banks and menu strings
- growth rates
- tileset collision / warp metadata

### Tier 3

- battle-effect categorization
- battle animations
- credits
- SGB presentation
- icon and tilemap tables

## Suggested normalized schema

A practical relational export could start with:

- `pokemon_species`
- `pokemon_base_stats`
- `pokemon_evolutions`
- `pokemon_levelup_moves`
- `pokemon_tmhm_compatibility`
- `pokedex_entries`
- `moves`
- `move_effects`
- `items`
- `mart_inventories`
- `trainer_classes`
- `trainer_parties`
- `trainer_party_members`
- `maps`
- `map_headers`
- `map_objects`
- `map_warps`
- `map_npcs`
- `wild_encounter_maps`
- `wild_encounter_slots`
- `npc_trades`
- `hidden_items`
- `hidden_events`
- `type_matchups`
- `growth_rates`
- `text_entries`

## Summary

If the goal is "what gameplay/content data can we import from this repo?", the answer is: **nearly all player-facing RPG content is present under `data/`**. The most important import domains are Pokemon, moves, items, trainers, maps, wild encounters, events, and text. The main challenge is not missing data; it is **parsing assembly tables, pointer tables, and include-based record layouts into normalized records**.
