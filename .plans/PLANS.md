# PLANS.md

## Objective

Refactor the Pokemon Red agent from a monolithic LLM-driven loop into a **mode-dispatched deterministic engine** where the LLM is responsible only for high-level objective planning and strategic decisions, while all low-level control (navigation, battle execution, menu traversal, dialogue advancement, map transitions) is handled by deterministic controllers.

The target end state: the agent can play through the game with the LLM called only every ~20-50 turns for objective updates, rather than every turn for action selection.

## Non-goals

- Optimal play (speed, EV training, perfect team composition)
- Supporting ROMs other than Pokemon Red (English)
- Real-time Twitch overlay features (can be layered later)
- Completing the Pokedex
- Post-game content
- Multiplayer features
- Full Gen 1 battle simulator accuracy (good enough > perfect)
- Importing ALL pokered data (only import what materially improves reliability)

## Current problems to fix

### P0 — Architectural

1. **engine.py is a 3,420-line god object.** It contains the turn loop, candidate generation, candidate compilation, objective planning, connector synthesis, navigation goal management, interrupt handling, bootstrap logic, and more. No clear separation of concerns.

2. **No game mode dispatcher.** Battle, menu, dialogue, and overworld all flow through the same `_resolve_turn_plan()` → `_build_candidate_steps()` → `_plan_candidates()` pipeline. The candidate system is the wrong abstraction for deterministic game modes.

3. **The LLM picks actions from 4 candidates every turn.** This is wasteful: most turns have an obvious deterministic answer (advance dialogue, select best move, navigate to target). The LLM should only be consulted for *strategic* decisions.

4. **No deterministic dialogue handler.** Text advancement (PRESS_A to continue) goes through the candidate system. This is a 0-decision operation being routed through an LLM pipeline.

5. **Connector activation modes are guessed from heuristics.** `world_map.py` infers push/step_on/interact from boundary position and neighbor count. The ROM data has this information explicitly via warp tables and map object definitions.

### P1 — Brittleness

6. **Menu navigation depends on VRAM text decoding.** `menu_manager.py` reads tile graphics from `game_area` to decode menu labels. This is fragile — if the rendering changes or tiles are ambiguous, menu classification fails.

7. **Battle manager has no real strategy.** Move selection is power×type_effectiveness. No understanding of stat stages, status moves, switching cost, PP conservation, or required battles vs. optional wild encounters.

8. **Stuck recovery is hint injection.** When stuck, the system injects text hints into the LLM prompt. There is no deterministic recovery protocol (try alternate routes, backtrack, interact with nearby objects).

9. **Navigation goal state is complex and fragile.** `NavigationGoal` has 17 fields tracking failed candidates, failed connectors, failed sides, confirmation requirements. This is symptom of the navigation system not being authoritative enough to just work.

10. **Objective plan replanning triggers are numerous and hard to reason about.** 9 different replan reasons, each with special handling. Plans become stale frequently, causing unnecessary LLM calls.

### P2 — Missing Capabilities

11. **No shop/heal/service flow automation.** Visiting a Pokemon Center or Mart requires navigating a known dialogue tree. There is no deterministic handler for these.
12. **No trainer/NPC interaction metadata.** The agent doesn't know which NPCs are trainers, which are story-critical, which are optional.
13. **No warp metadata from ROM.** Warp destinations, activation methods, and required items are available in pokered data but not imported.
14. **No sign/interactable metadata.** The agent doesn't know what objects are interactable or what they do.

## Target architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GameLoop                                  │
│  (thin orchestrator: read state → dispatch → execute → observe)  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │ ObjectiveMgr  │    │         ModeDispatcher                │   │
│  │ (LLM-driven)  │───▶│  state.mode → handler.step(state)    │   │
│  │               │    │                                        │   │
│  │ Sets:         │    │  OVERWORLD → OverworldController       │   │
│  │ - current goal│    │  BATTLE    → BattleController          │   │
│  │ - strategy    │    │  TEXT      → DialogueController        │   │
│  │ - next map    │    │  MENU      → MenuController            │   │
│  └──────────────┘    │  CUTSCENE  → CutsceneController        │   │
│         ▲             └──────────────────────────────────────┘   │
│         │                            │                            │
│    (replan when                      ▼                            │
│     milestone changes,         ActionDecision                     │
│     strategy needed,                 │                            │
│     stuck > threshold)               ▼                            │
│                              ┌──────────────┐                    │
│                              │   Emulator     │                    │
│                              └──────────────┘                    │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    GameKnowledge                           │   │
│  │  (authoritative ROM data + runtime discoveries)            │   │
│  │  - static world graph (maps, warps, connections)           │   │
│  │  - warp metadata (destination, activation, facing)         │   │
│  │  - map objects (NPCs, signs, interactables)                │   │
│  │  - type chart, move data, Pokemon stats                    │   │
│  │  - runtime: blocked tiles, discovered state, visited       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Responsibility split

| Component | Deterministic | LLM |
|-----------|:---:|:---:|
| Overworld navigation (pathfinding, walking) | ✓ | |
| Map transitions (warps, connections) | ✓ | |
| Dialogue advancement (PRESS_A) | ✓ | |
| Yes/No dialogue choices (heal, shop) | ✓ | |
| Story-branching dialogue choices | | ✓ |
| Battle move selection | ✓ | |
| Battle strategy (switch, item, run) | ✓ (heuristic) | fallback |
| Menu navigation to target | ✓ | |
| Shop/heal/service flows | ✓ | |
| "What to do next" (objective) | | ✓ |
| "Which route/strategy" (planning) | | ✓ |
| Stuck recovery (deterministic first) | ✓ | escalation |

## Required invariants

1. **Single authoritative game state.** `StructuredGameState` is the only source of truth. No shadow state in controllers.
2. **Mode dispatcher is exhaustive.** Every `GameMode` value has a handler. No mode falls through to LLM by default.
3. **Controllers are stateless across modes.** A controller may have internal state within its mode, but when the mode changes, the previous controller's state is irrelevant.
4. **LLM is never in the action loop.** The LLM sets objectives. Controllers execute toward objectives deterministically. The LLM is consulted again only when: (a) objective changes, (b) strategy decision needed, (c) stuck recovery escalation.
5. **Navigation is authoritative.** Given a target (x, y, map), the navigation system either produces a path or reports "unreachable" with a reason. No guessing.
6. **Warp activation is data-driven.** Connector activation modes come from imported ROM data, not heuristic inference.
7. **Recovery is deterministic first.** Try: alternate path → interact nearby → backtrack → random walk → THEN escalate to LLM.

## Dependency graph

```
Phase 1: Mode Dispatcher + Controller Interfaces
    │
    ├── Phase 2: Deterministic Controllers (dialogue, battle, menu)
    │       │
    │       └── Phase 3: Overworld Controller + Navigation Cleanup
    │               │
    │               └── Phase 4: ROM Data Import (warps, objects, type chart)
    │                       │
    │                       └── Phase 5: Objective Manager (LLM planning)
    │                               │
    │                               └── Phase 6: Recovery + Stuck Handling
    │                                       │
    │                                       └── Phase 7: Integration + engine.py Decomposition
```

Phases 2 and 3 can partially overlap. Phase 4 can start as soon as Phase 1 is done.

## Phased implementation plan

### Phase 1: Mode Dispatcher + Controller Interfaces

**Goal:** Establish the routing layer that replaces engine.py's monolithic `_resolve_turn_plan()`.

**Why:** Everything else depends on having a clean dispatch point. Without this, new controllers can't be plugged in.

**Code changes:**
- Create `src/pokemon_agent/agent/mode_dispatcher.py` with `ModeDispatcher` class
- Define `Controller` protocol: `step(state: StructuredGameState, context: TurnContext) -> PlanningResult`
- Extract `PlanningResult` into a shared module so controllers can return the existing rich planning payload without circular imports
- Create stub controllers for each mode that delegate to existing logic
- Wire `ModeDispatcher` into `engine.py` as the primary non-bootstrap planning path while keeping executor retries and live-state refresh in place

**Files affected:**
- NEW: `src/pokemon_agent/agent/mode_dispatcher.py`
- NEW: `src/pokemon_agent/agent/controllers/protocol.py`
- NEW: `src/pokemon_agent/agent/planning_types.py`
- MODIFY: `src/pokemon_agent/agent/engine.py` (route through dispatcher)

**Interfaces:**
```python
class Controller(Protocol):
    def step(self, state: StructuredGameState, context: TurnContext) -> PlanningResult: ...
    def reset(self) -> None: ...

class TurnContext:
    objective: Objective | None
    navigation_target: NavigationTarget | None
    stuck_score: int
    turn_index: int
```

**Acceptance criteria:**
- Dispatcher routes to correct stub controller based on effective mode (`battle_state`/`menu_open`/`text_box_open` before raw `state.mode`)
- Bootstrap remains engine-owned, but all normal non-executor planning uses the dispatcher
- Scoped dispatcher/engine verification passes; known unrelated reds remain out of scope for this milestone

### Phase 2: Deterministic Controllers

**Goal:** Replace candidate-based LLM selection for dialogue, battle, and menu with deterministic controllers.

**Why:** These modes have known-correct behavior that never needs LLM involvement. Routing them through candidates wastes tokens and introduces failure modes.

**Code changes:**
- `DialogueController`: PRESS_A for text advancement, deterministic yes/no for known patterns, flag unknown choices for LLM
- `BattleController`: best-move selection via type chart + power, switching heuristic, run-from-wild logic
- `MenuController`: deterministic menu navigation using cursor position + target item

**Files affected:**
- NEW: `src/pokemon_agent/agent/controllers/dialogue.py`
- NEW: `src/pokemon_agent/agent/controllers/battle.py`
- NEW: `src/pokemon_agent/agent/controllers/menu.py`
- MODIFY: `src/pokemon_agent/agent/battle_manager.py` (extract logic, may be deleted)
- MODIFY: `src/pokemon_agent/agent/menu_manager.py` (extract logic, may be deleted)

**Acceptance criteria:**
- Dialogue controller advances text without LLM
- Battle controller selects moves without LLM
- Menu controller navigates to target without LLM
- All 54+ existing tests pass

### Phase 3: Overworld Controller + Navigation Cleanup

**Goal:** Unify the 4 overlapping navigation systems into a single overworld controller with clear intra-map and inter-map pathfinding.

**Why:** Currently navigation is split across `executor.py`, `navigation.py`, `world_map.py`, `world_graph.py`, and `map_connections.py`. The executor duplicates navigation logic. The overworld controller should own all of it.

**Code changes:**
- `OverworldController`: given a navigation target (map, x, y), produce the next move
- Merge executor's navigate/interact/connector logic into overworld controller
- Single `Navigator` service that combines intra-map BFS + inter-map routing
- Remove `map_connections.py` wrapper (use world_graph directly)

**Files affected:**
- NEW: `src/pokemon_agent/agent/controllers/overworld.py`
- NEW: `src/pokemon_agent/agent/navigator.py` (unified navigation service)
- MODIFY: `src/pokemon_agent/agent/executor.py` (most logic moves to overworld controller)
- MODIFY: `src/pokemon_agent/agent/navigation.py` (becomes pure BFS utility)
- DELETE or THIN: `src/pokemon_agent/data/map_connections.py`

**Acceptance criteria:**
- Overworld controller can navigate to any reachable tile on current map
- Overworld controller can navigate between maps using known warps/connections
- Blocked tile handling works without separate TTL tracking in engine
- Executor.py is either deleted or reduced to a thin task-step adapter

### Phase 4: ROM Data Import

**Goal:** Import warp metadata, map objects (signs, NPCs, interactables), and type chart from pokered data to replace heuristic inference.

**Why:** Connector activation modes are currently guessed. NPC positions and types are unknown. Type chart is incomplete. This data exists in the ROM disassembly and should be authoritative.

**Link:** https://github.com/pret/pokered/tree/master/data

**Code changes:**
- Extend `scripts/import_pret_world_graph.py` to also import:
  - Warp metadata (destination, activation requirements)
  - Map object data (NPC positions, sign positions, interactable tiles)
  - Full Gen 1 type chart
- Store as JSON in `src/pokemon_agent/generated/`
- Create `GameKnowledge` service that loads and indexes all static data

**Files affected:**
- MODIFY: `scripts/import_pret_world_graph.py`
- NEW: `src/pokemon_agent/generated/map_objects.json`
- NEW: `src/pokemon_agent/generated/type_chart.json`
- NEW: `src/pokemon_agent/agent/game_knowledge.py`
- MODIFY: `src/pokemon_agent/agent/world_map.py` (use authoritative warp data)

**Data imports (from pokered_data_inventory.md):**
- `data/maps/objects/*.asm` → warp definitions, NPC positions, sign positions per map
- `data/types/type_matchups.asm` → full type effectiveness chart
- `data/maps/special_warps.asm` → fly destinations, dungeon warps
- `data/tilesets/warp_tile_ids.asm` → which tiles are warp tiles
- `data/tilesets/door_tile_ids.asm` → which tiles are doors

**Acceptance criteria:**
- Connector activation modes come from imported data, not heuristics
- Type chart is complete (all 15 Gen 1 types)
- Map object data available for at least first 20 story-critical maps
- `GameKnowledge` provides O(1) lookup for warp/NPC/sign by (map_id, x, y)

### Phase 5: Objective Manager

**Goal:** Replace the scattered objective planning logic in engine.py with a clean `ObjectiveManager` that owns all LLM interactions.

**Why:** Currently objective planning, replanning triggers, plan compilation, and plan staleness checking are spread across 500+ lines in engine.py. The LLM should be called through exactly one interface.

**Code changes:**
- `ObjectiveManager`: owns objective state, calls LLM for replanning, exposes `current_objective()` and `should_replan()`
- Move all objective-related code out of engine.py
- Simplify replan triggers to: milestone_changed, map_changed_to_unknown, stuck_escalation, plan_completed
- Remove `ObjectivePlanEnvelope` complexity — simplify to: `{goal: str, target_map: str, target_landmark: str, strategy: str}`

**Files affected:**
- NEW: `src/pokemon_agent/agent/objective_manager.py`
- MODIFY: `src/pokemon_agent/agent/engine.py` (remove ~500 lines of objective logic)
- MODIFY: `src/pokemon_agent/models/planner.py` (simplify plan models)
- MODIFY: `src/pokemon_agent/agent/context_manager.py` (simplify prompt building)

**Acceptance criteria:**
- LLM is called only through ObjectiveManager
- Replan triggers reduced from 9 to 4
- Objective model is simpler (no internal_plan/human_plan split)
- LLM calls reduced by 50%+ in typical play

### Phase 6: Recovery + Stuck Handling

**Goal:** Replace hint-injection stuck recovery with a deterministic recovery protocol that escalates to LLM only as last resort.

**Why:** Current recovery injects text hints into the prompt and hopes the LLM will pick a better action. This is unreliable. Recovery should be: try alternate path → interact nearby → backtrack → random walk → LLM escalation.

**Code changes:**
- `RecoveryController`: deterministic recovery protocol with escalation tiers
- Integrate with overworld controller (recovery replaces current navigation target)
- Remove stuck hint injection from context_manager
- Simplify stuck_detector to just track score + trigger recovery

**Files affected:**
- NEW: `src/pokemon_agent/agent/controllers/recovery.py`
- MODIFY: `src/pokemon_agent/agent/stuck_detector.py` (simplify)
- MODIFY: `src/pokemon_agent/agent/context_manager.py` (remove stuck hints)
- MODIFY: `src/pokemon_agent/agent/engine.py` (wire recovery)

**Acceptance criteria:**
- Stuck score > threshold triggers deterministic recovery before LLM
- Recovery tries 3 deterministic strategies before LLM escalation
- No recovery hint injection into prompts
- Stuck loops resolve within 10 turns in common scenarios

### Phase 7: Integration + engine.py Decomposition

**Goal:** Complete the decomposition of engine.py from 3,420 lines to <500 lines. Remove dead code, unused candidate types, and the old planning pipeline.

**Why:** After phases 1-6, most of engine.py's logic will live in controllers and managers. The remaining engine should be a thin orchestrator.

**Code changes:**
- engine.py becomes: read state → dispatch to controller → execute → observe → update memory
- Remove `_build_candidate_steps()`, `_plan_candidates()`, `_compile_candidate()` and all candidate infrastructure
- Remove `_candidate_runtime` dict and `CandidateRuntime`
- Remove connector synthesis (replaced by ROM data)
- Remove navigation goal management (owned by overworld controller)
- Clean up checkpoint save/load to reflect new component structure

**Files affected:**
- MODIFY: `src/pokemon_agent/agent/engine.py` (massive reduction)
- DELETE: old candidate-related code paths
- MODIFY: `src/pokemon_agent/models/planner.py` (remove candidate models if unused)
- MODIFY: checkpoint format (migration)

**Acceptance criteria:**
- engine.py is under 500 lines
- No candidate-based planning pipeline remains
- All existing test scenarios still pass
- Checkpoint format is documented and migration works

## Detailed task list

### Phase 1 Tasks

| ID | Title | Exact Change | Files | Deps | Done When | Tests |
|----|-------|-------------|-------|------|-----------|-------|
| 1.1 | Define Controller protocol | Create Protocol class with step() and reset() | controllers/protocol.py | — | Protocol importable, type-checks | Type check passes |
| 1.2 | Define TurnContext | Create TurnContext dataclass | controllers/protocol.py | — | Dataclass importable | Unit test for construction |
| 1.3 | Create ModeDispatcher | Route state.mode to controller, return ActionDecision | mode_dispatcher.py | 1.1, 1.2 | Dispatcher routes all 6 modes | Unit tests per mode |
| 1.4 | Create stub controllers | One stub per mode, delegates to existing engine logic | controllers/*.py | 1.1 | Stubs exist and return actions | Integration test |
| 1.5 | Wire dispatcher into engine | Replace _resolve_turn_plan branching with dispatcher call | engine.py | 1.3, 1.4 | Existing tests pass | Full test suite |

### Phase 2 Tasks

| ID | Title | Exact Change | Files | Deps | Done When | Tests |
|----|-------|-------------|-------|------|-----------|-------|
| 2.1 | DialogueController | PRESS_A for text, deterministic yes/no | controllers/dialogue.py | 1.5 | Advances all text without LLM | Unit tests for yes/no patterns |
| 2.2 | BattleController | Type-chart move selection, switch/run heuristics | controllers/battle.py | 1.5 | Selects moves without LLM | Unit tests vs battle_manager parity |
| 2.3 | MenuController | Cursor-based menu navigation | controllers/menu.py | 1.5 | Navigates menus without LLM | Unit tests for menu types |
| 2.4 | CutsceneController | PRESS_A or wait | controllers/cutscene.py | 1.5 | Handles cutscenes | Trivial test |
| 2.5 | Remove candidate path for modes | Remove _build_candidate_steps branches for battle/menu/text | engine.py | 2.1-2.4 | Candidate building only for overworld | Test suite passes |

### Phase 3 Tasks

| ID | Title | Exact Change | Files | Deps | Done When | Tests |
|----|-------|-------------|-------|------|-----------|-------|
| 3.1 | Create Navigator service | Unified intra-map BFS + inter-map routing | navigator.py | 1.5 | Navigator returns full path or error | Path tests |
| 3.2 | OverworldController | Navigate to target, handle interactions, handle warps | controllers/overworld.py | 3.1 | Walks to targets, enters warps | Navigation scenario tests |
| 3.3 | Merge executor into overworld | Move task execution logic from executor into overworld controller | overworld.py, executor.py | 3.2 | Executor.py deleted or minimal | All nav tests pass |
| 3.4 | Clean up blocked tile tracking | Single source of truth for blocked tiles in navigator | navigator.py, engine.py | 3.1 | No duplicate blocked tile tracking | Blocker regression tests |
| 3.5 | Delete map_connections.py | Replace all usages with direct world_graph calls | map_connections.py, imports | 3.1 | File deleted, no import errors | Import check |

### Phase 4 Tasks

| ID | Title | Exact Change | Files | Deps | Done When | Tests |
|----|-------|-------------|-------|------|-----------|-------|
| 4.1 | Import map object data | Parse data/maps/objects/*.asm → JSON (warps, NPCs, signs) | import script, generated/ | — | JSON generated for all maps | Spot-check 10 maps |
| 4.2 | Import type chart | Parse data/types/type_matchups.asm → JSON | import script, generated/ | — | All 15×15 matchups present | Compare vs hardcoded chart |
| 4.3 | Create GameKnowledge service | Load + index all generated JSON | game_knowledge.py | 4.1, 4.2 | O(1) lookup by (map, x, y) | Unit tests |
| 4.4 | Replace heuristic connector activation | Use warp metadata instead of guessing push/step_on | world_map.py | 4.3 | No heuristic activation mode | Compare with current behavior |
| 4.5 | Wire type chart into battle controller | Use imported chart instead of hardcoded | controllers/battle.py | 4.2, 2.2 | Full chart coverage | Type matchup tests |

### Phase 5 Tasks

| ID | Title | Exact Change | Files | Deps | Done When | Tests |
|----|-------|-------------|-------|------|-----------|-------|
| 5.1 | Create ObjectiveManager | Own objective state and LLM calls | objective_manager.py | 1.5 | Single LLM interface | Unit tests |
| 5.2 | Simplify objective model | Flatten ObjectivePlanEnvelope to simple goal struct | models/planner.py | 5.1 | Simpler model, fewer fields | Schema tests |
| 5.3 | Reduce replan triggers | 9 → 4 replan reasons | objective_manager.py | 5.1 | Only 4 trigger paths | Replan tests |
| 5.4 | Move objective code out of engine | Extract ~500 lines | engine.py | 5.1 | engine.py shorter | Existing tests |

### Phase 6 Tasks

| ID | Title | Exact Change | Files | Deps | Done When | Tests |
|----|-------|-------------|-------|------|-----------|-------|
| 6.1 | RecoveryController | Deterministic escalation tiers | controllers/recovery.py | 3.2 | 3 tiers before LLM | Stuck scenario tests |
| 6.2 | Simplify stuck_detector | Score only, no hint generation | stuck_detector.py | 6.1 | Under 60 lines | Unit tests |
| 6.3 | Remove stuck hints from context | No more recovery hint injection | context_manager.py | 6.1 | Prompts have no stuck hints | Prompt tests |

### Phase 7 Tasks

| ID | Title | Exact Change | Files | Deps | Done When | Tests |
|----|-------|-------------|-------|------|-----------|-------|
| 7.1 | Remove candidate infrastructure | Delete _build_candidate_steps, _plan_candidates, _compile_candidate | engine.py | 2.5, 3.3, 5.4 | No candidate code in engine | Compile check |
| 7.2 | Remove CandidateRuntime | Delete class and dict | engine.py, planner.py | 7.1 | No CandidateRuntime imports | Import check |
| 7.3 | Simplify checkpoint format | Reflect new component structure | engine.py | 7.1 | Checkpoint works with new structure | Save/load test |
| 7.4 | Final engine.py cleanup | Remove dead code, comments, unused imports | engine.py | 7.1-7.3 | Under 500 lines | Full test suite |

## Data model changes

### New: `TurnContext`
- **Purpose:** Immutable per-turn context passed to controllers
- **Owner:** GameLoop (engine.py)
- **Source of truth:** Composed from StructuredGameState + ObjectiveManager output
- **Fields:** objective, navigation_target, stuck_score, turn_index, previous_action, previous_progress

### New: `NavigationTarget`
- **Purpose:** Where to go and why
- **Owner:** ObjectiveManager (sets it) → OverworldController (consumes it)
- **Source of truth:** Derived from current objective + world graph
- **Fields:** map_name, x, y, reason, landmark_id (optional)

### Simplified: Objective (replaces ObjectivePlanEnvelope)
- **Purpose:** What the agent is trying to achieve
- **Owner:** ObjectiveManager
- **Source of truth:** LLM output, validated against milestones
- **Fields:** goal_text, target_map, target_landmark, strategy, milestone_id, confidence

### Removed: CandidateNextStep, CandidateRuntime, PlannerDecision
- **Reason:** Controllers produce actions directly; no candidate selection needed

### Retained: StructuredGameState, Task (simplified), ActionDecision, MemoryState

## Runtime loop changes

### Current loop (engine.py:run_turn)
```
read state → _resolve_turn_plan [8 retries, candidates, LLM] → validate → execute → observe → update memory
```

### Target loop
```
read state
  → mode_dispatcher.dispatch(state)
    → controller.step(state, context) → ActionDecision
  → execute action
  → read new state
  → detect progress
  → update memory
  → check if objective needs replan (ObjectiveManager)
  → check if stuck (trigger recovery controller if threshold)
```

Key differences:
- No retry loop inside planning (controllers are deterministic, they always return an action)
- No candidate generation or LLM selection
- Objective checking happens AFTER the turn, not before
- Recovery is a controller, not a prompt injection

## Recovery and retry policy

### Tier 1: Controller-level (immediate)
- Move blocked → try next BFS step, or alternate approach tile
- Menu navigation failed → re-read cursor position, try again
- Dialogue not advancing → repeat PRESS_A (up to 3x)

### Tier 2: Overworld recovery (stuck_score 2-4)
1. Try alternate path to same target (avoid recently blocked tiles)
2. Interact with nearest unvisited interactable
3. Backtrack to last known-good position

### Tier 3: Navigation recovery (stuck_score 5-7)
4. Walk to nearest unexplored boundary
5. Random walk (4 random moves)

### Tier 4: LLM escalation (stuck_score 8+)
6. Call LLM with full context: "I'm stuck at (x,y) on map M. I've tried A, B, C. What should I do?"
7. Reset stuck score on any progress

### Reset conditions:
- Map change: reset to 0
- Major progress: reset to 0
- Mode change: reset to 0
- Minor progress: decrement by 1

## Test and evaluation plan

### Unit tests
- Controller protocol compliance (each controller returns valid ActionDecision)
- DialogueController: text advancement, yes/no patterns, unknown choice flagging
- BattleController: move selection vs type chart, switching, running
- MenuController: cursor navigation, menu type detection
- OverworldController: pathfinding, warp entry, interaction
- Navigator: intra-map BFS, inter-map routing, blocked tile handling
- ObjectiveManager: replan triggers, objective simplification
- GameKnowledge: data loading, lookup correctness

### Integration tests
- Full turn loop with mock emulator (existing test infrastructure)
- Mode transition sequences (overworld → battle → overworld)
- Multi-map navigation (walk through 3+ maps)
- Bootstrap → first controllable state → first objective

### Scenario/replay tests
- Replay `.sessions/game` checkpoint and verify correct action selection
- Replay `.sessions/pathfinding` and verify route computation

### Long-horizon reliability tests
- Mock 100-turn overworld sequence: no stuck loops
- Mock battle sequence: all moves produce valid actions
- Mock menu sequence: navigates to target in ≤ 5 turns

### Metrics
- LLM calls per 100 turns (target: < 10)
- Stuck score > threshold events per 100 turns (target: < 3)
- engine.py line count (target: < 500)
- Mean turns to resolve stuck (target: < 8)

## Priority order

1. **Phase 1** — Mode dispatcher (unlocks everything else)
2. **Phase 2** — Dialogue + Battle + Menu controllers (biggest LLM call reduction)
3. **Phase 4** — ROM data import (can partially overlap with Phase 2)
4. **Phase 3** — Overworld controller + navigation cleanup
5. **Phase 5** — Objective manager
6. **Phase 6** — Recovery
7. **Phase 7** — Final cleanup

## Fastest high-leverage refactor

**If you can only do one thing:** Create `DialogueController` and `BattleController` and wire them into the turn loop via a simple mode check in `_resolve_turn_plan()`. Skip the full dispatcher for now — just add:

```python
if state.mode == GameMode.TEXT:
    return state, self._dialogue_controller.step(state)
if state.mode == GameMode.BATTLE:
    return state, self._battle_controller.step(state)
```

This eliminates LLM calls for the two most frequent non-overworld modes and can be done in < 200 lines of new code.

## Deferred work

- Pokemon team optimization (EV training, move tutors)
- Safari Zone special mechanics
- Slot machine / Game Corner automation
- Silph Co. card key door puzzle automation
- Pokemon Tower ghost encounters (require Silph Scope)
- Full trainer party import and pre-battle strategy
- Fly/teleport as navigation shortcuts
- PC box management
- Evolution handling
- TM/HM teaching optimization
- Item management (what to buy, when)
- Money management
- Level-up move learning choices

## Open questions

1. **Mock adapter coverage:** Does the mock adapter need to be updated for new controllers, or can controllers be tested with direct state construction?
2. **Checkpoint migration:** Should old checkpoints be forward-compatible, or is a clean break acceptable?
3. **Menu VRAM decoding:** Should menu_manager's text decoding be kept as a fallback, or should all menu navigation use cursor position + RAM state?
4. **Battle accuracy:** Is the simplified type chart + power heuristic sufficient, or do Gen 1 special mechanics (badge boosts, X Accuracy + OHKO, etc.) matter?
5. **World graph completeness:** Are there maps in the ROM that are missing from `world_graph.json`? (Some have `routing_enabled=false`)
6. **Sign text import:** Is it worth importing sign text from pokered so the agent knows what signs say without reading VRAM?

## Progress log

*(initialized empty — update during implementation)*

## Decision log

*(initialized empty — update during implementation)*

## Discoveries / surprises log

*(initialized empty — update during implementation)*

## Appendix: file-by-file change map

| File | Phase | Change |
|------|-------|--------|
| `agent/engine.py` | 1,2,3,5,7 | Progressive decomposition from 3420→<500 lines |
| `agent/executor.py` | 3 | Logic moves to overworld controller; delete or thin |
| `agent/navigation.py` | 3 | Becomes pure BFS utility; lose NavigationGrid class coupling |
| `agent/world_map.py` | 3,4 | Use authoritative warp data; simplify connector discovery |
| `agent/context_manager.py` | 5,6 | Simplify prompts; remove stuck hints; remove candidate serialization |
| `agent/battle_manager.py` | 2 | Logic moves to BattleController; delete |
| `agent/menu_manager.py` | 2 | Logic moves to MenuController; delete |
| `agent/memory_manager.py` | 5 | Simplify goal updates; remove candidate-related event tracking |
| `agent/progress.py` | — | Mostly unchanged |
| `agent/stuck_detector.py` | 6 | Simplify to score-only |
| `agent/validator.py` | 7 | Simplify or delete (controllers validate internally) |
| `agent/llm_client.py` | 5 | Unchanged (used by ObjectiveManager) |
| `agent/mode_dispatcher.py` | 1 | NEW: mode routing |
| `agent/controllers/protocol.py` | 1 | NEW: Controller protocol + TurnContext |
| `agent/controllers/dialogue.py` | 2 | NEW: deterministic dialogue |
| `agent/controllers/battle.py` | 2 | NEW: deterministic battle |
| `agent/controllers/menu.py` | 2 | NEW: deterministic menu |
| `agent/controllers/cutscene.py` | 2 | NEW: trivial cutscene handler |
| `agent/controllers/overworld.py` | 3 | NEW: navigation + interaction |
| `agent/controllers/recovery.py` | 6 | NEW: deterministic recovery |
| `agent/navigator.py` | 3 | NEW: unified navigation service |
| `agent/game_knowledge.py` | 4 | NEW: static ROM data index |
| `agent/objective_manager.py` | 5 | NEW: LLM objective planning |
| `models/planner.py` | 5,7 | Simplify; remove CandidateNextStep if unused |
| `models/action.py` | — | Mostly unchanged |
| `models/state.py` | — | Mostly unchanged |
| `models/memory.py` | 3,5 | Simplify NavigationGoal; simplify objective plan |
| `data/map_connections.py` | 3 | DELETE: replaced by direct world_graph usage |
| `data/walkthrough.py` | — | Unchanged |
| `data/map_names.py` | — | Unchanged |
| `navigation/world_graph.py` | — | Unchanged (core data loader) |
| `generated/*.json` | 4 | NEW: map_objects.json, type_chart.json |
| `scripts/import_pret_world_graph.py` | 4 | Extend to import map objects + type chart |
| `config.py` | — | Mostly unchanged |
| `main.py` | 1 | Minor: pass dispatcher to runner |
