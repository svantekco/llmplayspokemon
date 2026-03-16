# 05-objective-manager.md

## Objective

Extract all LLM objective planning logic from engine.py into a standalone `ObjectiveManager` that owns the LLM interface, reduces replan triggers from 9 to 4, and simplifies the objective model.

## Why this milestone exists

Currently ~500 lines of engine.py handle objective planning: replan trigger detection, LLM prompt building, response parsing, plan compilation, staleness checking. This is tangled with the turn loop and candidate system. The LLM should be called through exactly one clean interface.

## Scope

- Create `ObjectiveManager` that owns objective state and LLM calls
- Simplify `ObjectivePlanEnvelope` to a flat `Objective` struct
- Reduce replan triggers from 9 to 4: milestone_changed, map_to_unknown, stuck_escalation, plan_completed
- Move all objective-related code out of engine.py
- Simplify context_manager prompt building (no more candidate serialization for objective prompt)

## Non-goals

- Changing the LLM model or API
- Changing the milestone system
- Adding new objective types
- Multi-step objective chaining

## Current code to inspect first

- `src/pokemon_agent/agent/engine.py`: `_ensure_objective_plan()` (lines 588-625), `_objective_plan_replan_reason()` (lines 721-748), `_build_objective_snapshot()`, `_parse_objective_plan()`, `_compile_objective_plan()`
- `src/pokemon_agent/models/planner.py`: `ObjectivePlanEnvelope`, `HumanObjectivePlan`, `InternalObjectivePlan`
- `src/pokemon_agent/agent/context_manager.py`: `build_objective_snapshot()`

## Concrete code changes

### 1. Simplified Objective model

Replace `ObjectivePlanEnvelope` (with `HumanObjectivePlan` + `InternalObjectivePlan`) with:

```python
@dataclass
class Objective:
    goal: str                    # "Defeat Brock in Pewter Gym"
    target_map: str | None       # "Pewter Gym"
    target_landmark: str | None  # "gym_pewter"
    strategy: str                # "Navigate to Pewter City, then enter gym"
    milestone_id: str | None     # "gym1_brock"
    confidence: float = 0.8
    generated_at_step: int = 0
```

### 2. ObjectiveManager

```python
class ObjectiveManager:
    def __init__(self, llm_client, context_manager, walkthrough):
        self._llm = llm_client
        self._ctx = context_manager
        self._walkthrough = walkthrough
        self._current: Objective | None = None
        self._last_milestone_id: str | None = None
        self._last_replan_step: int = 0

    def current_objective(self) -> Objective | None:
        return self._current

    def should_replan(self, state, stuck_score, turn_index) -> bool:
        if self._current is None:
            return True
        milestone = self._walkthrough.get_current_milestone(state.story_flags, ...)
        if milestone.id != self._last_milestone_id:
            return True  # milestone changed
        if stuck_score >= 8 and turn_index - self._last_replan_step >= 10:
            return True  # stuck escalation (with cooldown)
        return False

    def replan(self, state, memory, stuck_score) -> Objective:
        # Build prompt, call LLM, parse response
        ...
        self._current = parsed_objective
        return self._current

    def mark_completed(self) -> None:
        self._current = None

    def navigation_target(self) -> NavigationTarget | None:
        if self._current is None:
            return None
        return NavigationTarget(
            map_name=self._current.target_map,
            landmark_id=self._current.target_landmark,
            reason=self._current.goal,
        )
```

### 3. Engine.py cleanup

Remove from engine.py:
- `_ensure_objective_plan()` and all helpers
- `_objective_plan_replan_reason()`
- `_build_objective_snapshot()`
- `_parse_objective_plan()`
- `_compile_objective_plan()`
- `_objective_plan_messages` and related state

Replace with:
```python
# In run_turn(), after action execution:
if self.objective_manager.should_replan(after, stuck_state.score, turn_index):
    self.objective_manager.replan(after, self.memory, stuck_state.score)
```

## Interfaces and contracts

### ObjectiveManager
- `current_objective()` returns None if no objective set (causes exploration mode)
- `should_replan()` is cheap (no LLM call)
- `replan()` calls LLM exactly once, returns new objective
- LLM failure → returns default objective from milestone data (no crash)
- `navigation_target()` derives target from objective (controller-facing)

## Data model changes

### Simplified: Objective (replaces ObjectivePlanEnvelope)
See struct above. Flat, no nested plans.

### Removed (eventually):
- `HumanObjectivePlan`
- `InternalObjectivePlan`
- `ObjectivePlanEnvelope`
- `ObjectivePlanStatus`

### Modified:
- `LongTermKnowledge.objective_plan` → `LongTermKnowledge.objective`
- `MemoryState` checkpoint format updated

## Migration plan

1. Create ObjectiveManager with tests
2. Wire into engine.py alongside existing code (dual-path)
3. Verify identical behavior
4. Remove old objective code from engine.py
5. Update checkpoint format

## Tests and evaluation

- Replan triggers: only 4 conditions cause replan
- LLM called at most once per replan
- LLM failure produces fallback objective
- Milestone change detected correctly
- Stuck escalation respects cooldown
- Navigation target derived correctly

## Risks and edge cases

- **Checkpoint format change.** Old checkpoints store ObjectivePlanEnvelope. Migration: treat old format as "no objective" (force replan).
- **LLM prompt simplification.** The current prompt is complex. Simpler prompt may produce worse objectives. Mitigate: test with real LLM before removing old path.

## Acceptance criteria

1. All LLM calls go through ObjectiveManager
2. Replan triggers reduced to 4
3. Objective model is a flat struct (no nested plans)
4. engine.py reduced by ~500 lines
5. LLM calls reduced (no more turn-level candidate selection for non-overworld)
6. All tests pass

## Rollback / fallback notes

- Old objective code in engine.py preserved as dead code until new path verified
- Checkpoint migration: old format triggers replan (safe)

## Ordered implementation checklist

1. **Step 5.1: Define simplified Objective model**
   - Action: Create flat Objective dataclass
   - Files: `models/planner.py`
   - Dependencies: none
   - Done: importable, frozen, serializable
   - Tests: construction + serialization

2. **Step 5.2: Create ObjectiveManager**
   - Action: Implement should_replan, replan, navigation_target
   - Files: `agent/objective_manager.py`
   - Dependencies: 5.1
   - Done: manages objective lifecycle
   - Tests: `test_objective_manager.py`

3. **Step 5.3: Wire into engine.py**
   - Action: Add ObjectiveManager to engine, call in run_turn
   - Files: `agent/engine.py`
   - Dependencies: 5.2
   - Done: objectives managed by ObjectiveManager
   - Tests: full test suite

4. **Step 5.4: Remove old objective code**
   - Action: Delete _ensure_objective_plan and related methods
   - Files: `agent/engine.py`
   - Dependencies: 5.3
   - Done: engine.py ~500 lines shorter
   - Tests: full test suite

5. **Step 5.5: Update checkpoint format**
   - Action: Save/load Objective instead of ObjectivePlanEnvelope
   - Files: `agent/engine.py`
   - Dependencies: 5.4
   - Done: checkpoints work with new format
   - Tests: save/load round-trip test
