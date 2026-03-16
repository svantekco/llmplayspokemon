# 07-integration-cleanup.md

## Objective

Complete the decomposition of `engine.py` from 3,420 lines to under 500 lines by removing all candidate infrastructure, dead code, and superseded logic.

## Why this milestone exists

After Phases 1-6, most of engine.py's logic lives in controllers, ObjectiveManager, Navigator, and RecoveryController. The remaining engine should be a thin orchestrator: read state → dispatch → execute → observe → update memory.

## Scope

- Remove candidate generation, selection, and compilation infrastructure
- Remove CandidateRuntime, CandidateNextStep usage from engine
- Remove connector synthesis (replaced by ROM data)
- Remove navigation goal management (owned by overworld controller)
- Simplify checkpoint save/load
- Clean up imports and dead code
- Delete `battle_manager.py` and `menu_manager.py` if fully superseded

## Non-goals

- Adding new features
- Changing the emulator interface
- Changing the memory model fundamentally

## Current code to inspect first

- `src/pokemon_agent/agent/engine.py`: everything
- `src/pokemon_agent/models/planner.py`: CandidateNextStep, CandidateRuntime, PlannerDecision

## Concrete code changes

### Remove from engine.py

- `_build_candidate_steps()` (~100 lines)
- `_plan_candidates()` (~100 lines)
- `_compile_candidate()` (~60 lines)
- `_compile_task_for_candidate()` (~40 lines)
- `_build_connector_candidates()` (~110 lines)
- `_build_static_connector_candidates()` (~100 lines)
- `_build_boundary_candidates()` (~80 lines)
- `_build_exploration_candidates()` (~60 lines)
- `_build_progression_candidates()` (~80 lines)
- `_synthesize_static_connector()` (~20 lines)
- `_candidate_runtime_for()` and `_candidate_runtime` dict
- All navigation goal management (~100 lines)
- Connector synthesis logic (~50 lines)
- Bootstrap special handling (if superseded by cutscene controller)
- Interaction cooldown tracking (move to overworld controller)

### Target engine.py structure (~400 lines)

```python
class ClosedLoopRunner:
    def __init__(self, emulator, dispatcher, objective_mgr, memory, progress, stuck):
        ...

    def run_turn(self, turn_index) -> TurnResult:
        state = self.emulator.get_structured_state()
        context = self._build_context(state, turn_index)

        action = self.dispatcher.dispatch(state, context)
        action = self._validate(action, state)

        self.emulator.execute_action(action)
        after = self.emulator.get_structured_state()

        progress = self.progress.compare(state, after)
        stuck = self.stuck.update(after, action, progress)

        events = self.memory.update_from_transition(state, after, action, progress, stuck)
        self.context_manager.record_turn(turn_index, action, after, progress, events, stuck)

        if self.objective_mgr.should_replan(after, stuck.score, turn_index):
            self.objective_mgr.replan(after, self.memory, stuck.score)

        return TurnResult(...)

    def save_checkpoint(self, path): ...
    def load_checkpoint(self, path): ...
    def _build_context(self, state, turn_index) -> TurnContext: ...
    def _validate(self, action, state) -> ActionDecision: ...
```

### Delete files

- `agent/battle_manager.py` (replaced by controllers/battle.py)
- `agent/menu_manager.py` (replaced by controllers/menu.py)
- `data/map_connections.py` (replaced by direct world_graph usage)
- `agent/executor.py` (replaced by controllers/overworld.py)

### Simplify models/planner.py

If CandidateNextStep and CandidateRuntime are no longer imported anywhere:
- Delete them
- Keep only Objective (simplified) and PlannerDecision (if still used by ObjectiveManager)

### Update checkpoint format

```json
{
  "version": 2,
  "completed_turns": 42,
  "memory": {},
  "stuck_state": {},
  "objective": {},
  "controller_states": {
    "overworld": {},
    "battle": {}
  }
}
```

## Interfaces and contracts

### ClosedLoopRunner (simplified)
- `run_turn()` → TurnResult (always succeeds, no retry loop)
- `save_checkpoint()` / `load_checkpoint()` (new format)
- No internal planning logic
- No candidate infrastructure

## Data model changes

### Removed
- CandidateNextStep (if unused)
- CandidateRuntime (if unused)
- PlannerDecision (if unused)
- NavigationGoal (replaced by NavigationTarget)

### Simplified
- TurnResult: fewer fields (no candidates, no raw_model_response for turn planning)

## Migration plan

1. Verify all controllers are wired and working
2. Remove candidate code paths one method at a time
3. After each removal, run tests
4. Delete superseded files
5. Update checkpoint format with migration
6. Final cleanup pass

## Tests and evaluation

- engine.py under 500 lines
- No candidate-related imports remain
- All 54+ tests pass
- Checkpoint save/load works with new format
- New turn loop produces same observable behavior

## Risks and edge cases

- **Missing edge cases in controllers.** Some candidate logic may handle cases that controllers don't yet. Mitigate: review each deleted method for edge cases before removal.
- **Checkpoint compatibility.** Old checkpoints won't load. Mitigate: version field + migration on load.

## Acceptance criteria

1. engine.py under 500 lines
2. No candidate infrastructure remains
3. battle_manager.py, menu_manager.py, executor.py, map_connections.py deleted
4. Checkpoint format v2 works
5. All tests pass
6. No dead imports

## Rollback / fallback notes

Keep git history. Each deleted method is recoverable from version control.

## Ordered implementation checklist

1. **Step 7.1: Remove candidate generation**
   - Action: Delete _build_candidate_steps and all sub-builders
   - Files: engine.py
   - Dependencies: Phases 2, 3 complete
   - Done: no candidate generation code
   - Tests: all tests pass

2. **Step 7.2: Remove candidate compilation**
   - Action: Delete _compile_candidate, _candidate_runtime_for, _candidate_runtime dict
   - Files: engine.py
   - Dependencies: 7.1
   - Done: no compilation code
   - Tests: all tests pass

3. **Step 7.3: Remove connector synthesis**
   - Action: Delete _synthesize_static_connector and related
   - Files: engine.py
   - Dependencies: Phase 4 complete
   - Done: no synthesis code
   - Tests: all tests pass

4. **Step 7.4: Delete superseded files**
   - Action: Delete battle_manager.py, menu_manager.py, executor.py, map_connections.py
   - Files: listed files
   - Dependencies: 7.1-7.3
   - Done: files deleted, no import errors
   - Tests: all tests pass

5. **Step 7.5: Clean up models/planner.py**
   - Action: Remove CandidateNextStep, CandidateRuntime if unused
   - Files: models/planner.py
   - Dependencies: 7.4
   - Done: only used models remain
   - Tests: import check

6. **Step 7.6: Update checkpoint format**
   - Action: New format with version field, migration for old format
   - Files: engine.py
   - Dependencies: 7.4
   - Done: save/load works
   - Tests: round-trip test

7. **Step 7.7: Final cleanup**
   - Action: Remove dead imports, unused variables, stale comments
   - Files: engine.py, all remaining files
   - Dependencies: 7.1-7.6
   - Done: engine.py < 500 lines, clean
   - Tests: full test suite + mypy + ruff
