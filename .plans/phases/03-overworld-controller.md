# 03-overworld-controller.md

## Objective

Unify overworld navigation into a single `OverworldController` backed by a `Navigator` service, replacing the current split across `executor.py`, `navigation.py`, `world_map.py`, and `map_connections.py`.

## Why this milestone exists

Navigation is the core gameplay loop. Currently it's fragmented across 4+ files with overlapping responsibilities:
- `executor.py` handles task-level navigation (NAVIGATE_TO, ENTER_CONNECTOR, etc.)
- `navigation.py` handles BFS pathfinding
- `world_map.py` handles connector discovery and inter-map BFS
- `map_connections.py` is a thin wrapper over world_graph
- `engine.py` handles navigation goal state, blocked tile TTLs, connector synthesis

This fragmentation causes:
- Duplicate blocked tile tracking (engine + executor + navigation grid)
- Unclear ownership of navigation state
- Complex connector activation state machines
- Navigation goals with 17 fields tracking every possible failure mode

## Scope

- Create `Navigator` service (unified intra-map + inter-map pathfinding)
- Create `OverworldController` that uses Navigator to produce single-step actions
- Consolidate blocked tile tracking into Navigator
- Absorb executor's navigate/interact/connector logic into OverworldController
- Delete or thin `executor.py`
- Delete `map_connections.py` wrapper

## Non-goals

- Changing how connectors are discovered (that stays in world_map.py)
- Importing new ROM data (Phase 4)
- Changing the world_graph loader
- Perfect pathfinding (BFS is sufficient)
- Fly/teleport navigation shortcuts

## Current code to inspect first

- `src/pokemon_agent/agent/executor.py`: full file, especially `_step_navigate_to()`, `_step_enter_connector()`, `_step_walk_boundary()`
- `src/pokemon_agent/agent/navigation.py`: `NavigationGrid`, `find_path()`, `build_navigation_snapshot_from_collision()`
- `src/pokemon_agent/agent/world_map.py`: `shortest_confirmed_path()`, `connectors_from_map()`, connector discovery
- `src/pokemon_agent/agent/engine.py`: `_build_connector_candidates()`, `_build_static_connector_candidates()`, `_synthesize_static_connector()`, navigation goal management, blocked tile TTLs
- `src/pokemon_agent/data/map_connections.py`: all functions

## Concrete code changes

### 1. Navigator service (`agent/navigator.py`)

Single class that owns all pathfinding state:

```python
class Navigator:
    def __init__(self, world_graph: WorldGraph, world_map_memory: WorldMapMemory):
        self._world_graph = world_graph
        self._world_map = world_map_memory
        self._blocked_tiles: dict[tuple[int, int], int] = {}  # (x,y) → expiry turn
        self._current_map_grid: NavigationGrid | None = None

    def update_grid(self, state: StructuredGameState) -> None:
        """Refresh local navigation grid from current state."""

    def find_local_path(self, from_xy, to_xy) -> list[ActionType] | None:
        """Intra-map BFS avoiding blocked tiles."""

    def find_route_to_map(self, from_map, to_map) -> list[DiscoveredConnector] | None:
        """Inter-map BFS over confirmed connectors, falling back to static graph."""

    def next_connector_toward(self, target_map) -> DiscoveredConnector | None:
        """First connector in route to target map."""

    def mark_blocked(self, x, y, expiry_turn) -> None:
        """Mark tile as temporarily blocked."""

    def prune_blocked(self, current_turn) -> None:
        """Remove expired blockers."""

    def approach_tile_for(self, connector) -> tuple[int, int] | None:
        """Best walkable tile adjacent to connector's activation point."""
```

### 2. OverworldController (`controllers/overworld.py`)

```python
class OverworldController:
    def __init__(self, navigator: Navigator, world_map: WorldMapMemory):
        self._navigator = navigator
        self._target: NavigationTarget | None = None
        self._phase: OverworldPhase = OverworldPhase.IDLE

    def step(self, state, context) -> ActionDecision:
        self._navigator.update_grid(state)
        target = context.navigation_target

        if target is None:
            return self._explore(state)

        # Same map: walk to target
        if self._on_target_map(state, target):
            return self._walk_to(state, target.x, target.y)

        # Different map: find route and enter next connector
        connector = self._navigator.next_connector_toward(target.map_name)
        if connector is not None:
            return self._enter_connector(state, connector)

        # No route: explore toward map boundary
        return self._walk_to_boundary(state, target)

    def _walk_to(self, state, target_x, target_y) -> ActionDecision:
        path = self._navigator.find_local_path((state.x, state.y), (target_x, target_y))
        if path:
            return ActionDecision(action=path[0], reason=f"walking to ({target_x},{target_y})")
        return self._handle_blocked(state)

    def _enter_connector(self, state, connector) -> ActionDecision:
        # Navigate to approach tile, then activate
        ...

    def _explore(self, state) -> ActionDecision:
        # No target: interact with nearby, or walk to boundary
        ...
```

### 3. Executor migration

Move task-execution logic from `executor.py` into `OverworldController`:
- `NAVIGATE_TO` → `_walk_to()`
- `NAVIGATE_ADJACENT` → `_walk_to()` with adjacency check + facing
- `INTERACT` → PRESS_A (inline)
- `ENTER_CONNECTOR` → `_enter_connector()`
- `WALK_BOUNDARY` → `_walk_to_boundary()`
- `PRESS_BUTTON` → inline (return the button action)

The key difference: the controller steps one action at a time and tracks its phase internally, rather than maintaining a separate `Task` object with retry state.

### 4. Delete `map_connections.py`

Replace all imports:
- `exits_from()` → `world_graph.neighbors()`
- `shortest_map_path()` → `world_graph.find_route()`
- `direction_toward()` → `world_graph.find_route()` first edge direction

### 5. Consolidate blocked tiles

Remove:
- `engine._temporary_blocked_tiles` and TTL tracking
- `executor._nav_grid` blocked tile tracking
- `engine._prune_temporary_blocked_tiles()`

Replace with:
- `Navigator._blocked_tiles` with turn-based expiry (single source of truth)

## Interfaces and contracts

### Navigator
- `find_local_path()` returns None if no path exists (not an error)
- `mark_blocked()` always succeeds (overwrites existing entry)
- Grid is refreshed each turn via `update_grid()` — never stale
- Blocked tiles expire automatically via `prune_blocked()`

### OverworldController
- Always returns an ActionDecision (never None)
- If no target and no exploration option: return PRESS_A as safe default
- Tracks internal phase (WALKING, ENTERING_CONNECTOR, EXPLORING) for multi-turn operations
- Reset clears phase and target

## Data model changes

### Simplified: `NavigationTarget` (defined in Phase 1)
Replaces the 17-field `NavigationGoal` in `models/memory.py`.

### Internal: `OverworldPhase`
```python
class OverworldPhase(Enum):
    IDLE = "idle"
    WALKING_TO_TARGET = "walking"
    APPROACHING_CONNECTOR = "approaching"
    ACTIVATING_CONNECTOR = "activating"
    EXPLORING_BOUNDARY = "exploring"
```

### Removed (from memory.py eventually):
- `NavigationGoal` (replaced by simpler NavigationTarget + controller phase)

## Migration plan

1. Create Navigator service with tests
2. Create OverworldController using Navigator
3. Wire OverworldController into dispatcher (replacing overworld stub)
4. Verify all navigation tests pass
5. Remove executor.py (or thin to adapter)
6. Remove map_connections.py
7. Remove blocked tile tracking from engine.py
8. Remove NavigationGoal from memory model (or simplify)

## Tests and evaluation

### Navigator tests
- Intra-map BFS finds shortest path
- Blocked tiles are avoided
- Blocked tiles expire after TTL
- Inter-map routing finds path through connectors
- No path returns None (not error)

### OverworldController tests
- Target on same map: walks toward target
- Target on different map: enters connector
- No target: explores
- Blocked path: tries alternate
- Connector entry: navigates to approach, then activates
- Reset clears state

### Integration tests
- Multi-map navigation through mock emulator
- Connector entry + map transition
- Blocked tile recovery

## Risks and edge cases

- **Connector activation regression.** The executor has complex push/interact fallback logic. The overworld controller needs to replicate this. Risk: missing edge cases.
- **Grid refresh timing.** If the grid is refreshed before the action is executed, the state may be stale. Mitigate: grid refresh happens at start of step(), which uses the latest state.
- **Multi-turn connector entry.** Entering a connector takes 2-3 turns (approach + activate). The controller needs phase tracking. Risk: mode changes mid-entry reset the phase.
- **Executor state in checkpoints.** Old checkpoints have executor state. Migration: ignore executor state in checkpoint, start fresh.

## Acceptance criteria

1. OverworldController navigates to any reachable tile on current map
2. OverworldController navigates between maps using connectors
3. Single source of truth for blocked tiles (Navigator)
4. executor.py deleted or reduced to < 50 lines
5. map_connections.py deleted
6. No blocked tile tracking in engine.py
7. All existing navigation tests pass
8. New tests for Navigator and OverworldController

## Rollback / fallback notes

This is the riskiest phase because it replaces the most complex existing logic. Mitigate:
- Keep executor.py until OverworldController passes all its tests
- Run both paths in parallel during development (compare outputs)
- Old checkpoint format support can be dropped (clean break)

## Ordered implementation checklist

1. **Step 3.1: Create Navigator with intra-map BFS**
   - Action: Create `navigator.py` with `update_grid()`, `find_local_path()`, blocked tile management
   - Files: `agent/navigator.py`
   - Dependencies: Phase 1
   - Done: BFS works on mock states
   - Tests: `test_navigator.py` with 8+ path cases

2. **Step 3.2: Add inter-map routing to Navigator**
   - Action: Add `find_route_to_map()`, `next_connector_toward()`, `approach_tile_for()`
   - Files: `agent/navigator.py`
   - Dependencies: 3.1
   - Done: can route through confirmed connectors
   - Tests: multi-map route tests

3. **Step 3.3: Create OverworldController**
   - Action: Implement step() with WALK/CONNECTOR/EXPLORE phases
   - Files: `controllers/overworld.py`
   - Dependencies: 3.2
   - Done: produces actions for all overworld scenarios
   - Tests: `test_overworld_controller.py`

4. **Step 3.4: Wire OverworldController into dispatcher**
   - Action: Replace overworld stub with real controller
   - Files: `agent/engine.py`
   - Dependencies: 3.3
   - Done: overworld uses new controller
   - Tests: full test suite passes

5. **Step 3.5: Remove executor.py**
   - Action: Delete or reduce to checkpoint compat shim
   - Files: `agent/executor.py`
   - Dependencies: 3.4
   - Done: no executor usage in engine
   - Tests: all tests pass without executor

6. **Step 3.6: Remove map_connections.py**
   - Action: Replace all imports with direct world_graph calls
   - Files: `data/map_connections.py`, all importers
   - Dependencies: 3.4
   - Done: file deleted, no import errors
   - Tests: import check + route tests

7. **Step 3.7: Consolidate blocked tile tracking**
   - Action: Remove engine._temporary_blocked_tiles, use Navigator only
   - Files: `agent/engine.py`, `agent/navigator.py`
   - Dependencies: 3.5
   - Done: single blocked tile source of truth
   - Tests: blocked tile expiry tests
