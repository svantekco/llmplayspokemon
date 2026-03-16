# 01-mode-dispatcher.md

## Objective

Create a `ModeDispatcher` that routes each game mode to a dedicated controller, replacing the monolithic `_resolve_turn_plan()` branching in `engine.py`.

## Why this milestone exists

Everything else in the refactor depends on having a clean dispatch point. Without it, new controllers can't be plugged in without touching the 3,420-line engine.py. The dispatcher is the structural prerequisite for all subsequent phases.

## Scope

- Define the `Controller` protocol (interface that all mode handlers implement)
- Define `TurnContext` (immutable per-turn context for controllers)
- Create `ModeDispatcher` class that maps `GameMode` → `Controller`
- Create **stub** controllers for each mode that wrap existing engine logic
- Extract `PlanningResult` into a shared module so controller returns can preserve current executor/planner metadata
- Wire the dispatcher into `engine.py` so normal planning delegates through it while the executor retry loop stays intact

## Non-goals

- Implementing real controller logic (stubs only in this phase)
- Changing any existing behavior (pure structural refactor)
- Removing candidate infrastructure yet
- Changing the LLM integration
- Modifying the checkpoint format

## Current code to inspect first

- `src/pokemon_agent/agent/engine.py`: `_resolve_turn_plan()` (lines 375-449), `_plan_action()` (lines 451-472), `_build_candidate_steps()` (lines 1053-1142)
- `src/pokemon_agent/models/state.py`: `GameMode` enum, `StructuredGameState`
- `src/pokemon_agent/models/action.py`: `ActionDecision`, `ActionType`
- `src/pokemon_agent/models/planner.py`: `Objective`, `ObjectiveTarget`

## Concrete code changes

### 1. Create `src/pokemon_agent/agent/controllers/__init__.py`
Empty init file for the controllers package.

### 2. Create `src/pokemon_agent/agent/controllers/protocol.py`

```python
from typing import Protocol
from dataclasses import dataclass
from pokemon_agent.models.state import StructuredGameState
from pokemon_agent.models.planner import Objective
from pokemon_agent.agent.planning_types import PlanningResult

@dataclass(frozen=True)
class NavigationTarget:
    map_name: str | None = None
    x: int | None = None
    y: int | None = None
    landmark_id: str | None = None
    reason: str = ""

@dataclass(frozen=True)
class TurnContext:
    objective: Objective | None = None
    navigation_target: NavigationTarget | None = None
    stuck_score: int = 0
    turn_index: int = 0
    previous_action: ActionDecision | None = None
    previous_progress: str | None = None  # classification string

class Controller(Protocol):
    def step(self, state: StructuredGameState, context: TurnContext) -> PlanningResult:
        """Produce the next planning result for this game mode."""
        ...

    def reset(self) -> None:
        """Reset internal state (called on mode transitions)."""
        ...
```

### 3. Create `src/pokemon_agent/agent/mode_dispatcher.py`

```python
class ModeDispatcher:
    def __init__(self, controllers: dict[GameMode, Controller]):
        self._controllers = controllers
        self._last_mode: GameMode | None = None

    def dispatch(self, state: StructuredGameState, context: TurnContext) -> ActionDecision:
        mode = self.effective_mode(state)
        if mode != self._last_mode and self._last_mode is not None:
            # Reset previous controller on mode change
            if self._last_mode in self._controllers:
                self._controllers[self._last_mode].reset()
        self._last_mode = mode
        controller = self._controllers.get(mode)
        if controller is None:
            raise ValueError(f"No controller registered for mode {mode}")
        return controller.step(state, context)
```

### 4. Create stub controllers

One file per mode (`controllers/stubs.py` or individual files), each wrapping existing engine logic by delegating back to the engine's existing methods through a callback or reference.

The stubs should use a `EngineCompat` interface so that during this phase, the stubs can call back into engine.py's existing methods:

```python
class StubOverworldController:
    def __init__(self, engine_plan_fn):
        self._plan = engine_plan_fn

    def step(self, state, context):
        return self._plan(state, context)  # delegates to existing engine compat leaf

    def reset(self):
        pass
```

### 5. Wire into engine.py

In `_resolve_turn_plan()`, add dispatcher as the primary path:

```python
def _resolve_turn_plan(self):
    state = self.emulator.get_structured_state()
    planning = self._plan_action(state)  # now builds context and dispatches
    return state, planning
```

Keep the existing executor stepping / retry / fallback path in `_resolve_turn_plan()`. Bootstrap also stays outside the dispatcher in this phase.

## Interfaces and contracts

### Controller protocol
- `step()` MUST always return a valid `PlanningResult`. For direct button actions, that means `PlanningResult(action=...)`.
- `step()` MUST NOT call the emulator directly. It receives state and returns an action.
- `reset()` is called when the mode changes away from this controller's mode.

### ModeDispatcher contract
- Every `GameMode` enum value MUST have a registered controller. Missing controller = ValueError.
- Mode transitions trigger `reset()` on the outgoing controller.
- Effective mode resolution is `BATTLE` > `MENU` > `TEXT` > raw `state.mode` to preserve current behavior.
- The dispatcher is stateless except for tracking the last mode (for reset triggering).

### TurnContext contract
- Immutable (frozen dataclass). Controllers must not modify it.
- Created fresh each turn by the engine.

## Data model changes

### New: `TurnContext` (see protocol.py above)
### New: `NavigationTarget` (see protocol.py above)
### New: shared `PlanningResult` module under `src/pokemon_agent/agent/`

No changes to existing models in this phase.

## Migration plan

1. Create controller package and protocol (no behavioral change)
2. Create stub controllers (no behavioral change)
3. Create dispatcher (no behavioral change)
4. Wire dispatcher into engine without changing the executor retry loop
5. Verify scoped dispatcher/engine tests with dispatcher path
6. Keep dispatcher as the default path; do not leave a runtime flag behind

## Tests and evaluation

### New unit tests
- `test_mode_dispatcher.py`:
  - Dispatcher routes to correct controller based on effective mode
  - Mode change triggers reset() on previous controller
  - Missing controller raises ValueError
  - UI flags override raw `state.mode`
- Stub-controller checks:
  - Verify stub controllers conform to Protocol
  - Verify step() returns `PlanningResult`
  - Verify reset() is callable

### Existing tests
- Scoped verification must pass with `PYTHONPATH=src`
- Run: `PYTHONPATH=src pytest tests/test_engine.py tests/test_executor.py tests/test_progress.py tests/test_validator.py tests/test_world_graph.py tests/test_mode_dispatcher.py -q`
- Known unrelated failures in `tests/test_context_manager.py`, `tests/test_menu_manager.py`, and `tests/test_walkthrough.py` are out of scope for this milestone

## Risks and edge cases

- **Stub controllers may not perfectly replicate engine behavior** during the transition. Mitigate by keeping old path as fallback and comparing outputs.
- **Mode detection may be wrong** (`state.mode` is heuristic). Mitigate by deriving effective mode from battle/menu/text flags before consulting `state.mode`.
- **UNKNOWN mode** still needs a controller. In this phase it intentionally reuses the existing non-interrupt planning path.
- **Engine re-entrancy**: stubs call back into engine methods. Avoid infinite loops by ensuring stubs only call leaf methods.

## Acceptance criteria

1. `ModeDispatcher` exists and routes all 6 `GameMode` values via effective-mode resolution
2. All stub controllers implement the `Controller` protocol
3. `TurnContext` and `NavigationTarget` are defined and importable
4. `engine.py` uses the dispatcher in `_plan_action()` / normal planning, while `_resolve_turn_plan()` keeps executor retries
5. Scoped dispatcher/engine verification passes
6. No change in observable behavior (pure structural refactor)

## Rollback / fallback notes

If the dispatcher introduces regressions:
- Rollback = remove dispatcher construction, restore direct `_plan_action()` branching
- No checkpoint/data model changes need to be reverted

## Ordered implementation checklist

1. **Step 1.1: Create controllers package**
   - Action: Create `src/pokemon_agent/agent/controllers/__init__.py` (empty)
   - Files: `controllers/__init__.py`
   - Dependencies: none
   - Done: package importable
   - Tests: `from pokemon_agent.agent.controllers import protocol` works

2. **Step 1.2: Define Controller protocol and TurnContext**
   - Action: Create `controllers/protocol.py` with Protocol, TurnContext, NavigationTarget
   - Files: `controllers/protocol.py`
   - Dependencies: 1.1
   - Done: types importable and type-check with mypy
   - Tests: `mypy src/pokemon_agent/agent/controllers/protocol.py`

3. **Step 1.3: Create ModeDispatcher**
   - Action: Create `mode_dispatcher.py` with dispatch() and mode-change reset logic
   - Files: `agent/mode_dispatcher.py`
   - Dependencies: 1.2
   - Done: dispatcher can be instantiated with controller dict
   - Tests: `test_mode_dispatcher.py` — routes correctly, resets on mode change

4. **Step 1.4: Create stub controllers**
   - Action: Create one stub per GameMode that delegates to existing engine methods
   - Files: `controllers/stubs.py` (or individual files)
   - Dependencies: 1.2
   - Done: stubs instantiable, step() returns `PlanningResult`
   - Tests: stub compliance tests

5. **Step 1.5: Wire dispatcher into engine.py**
   - Action: Add dispatcher construction in `__init__`, dispatch from `_plan_action()`, keep executor retry logic in `_resolve_turn_plan()`
   - Files: `agent/engine.py`
   - Dependencies: 1.3, 1.4
   - Done: dispatcher is the primary action source
   - Tests: scoped dispatcher/engine suite passes

6. **Step 1.6: Verify and remove fallback**
   - Action: Remove temporary scaffolding but leave dispatcher as the default path
   - Files: `agent/engine.py`
   - Dependencies: 1.5
   - Done: no dual-path code remains
   - Tests: full test suite + manual checkpoint test
