# 06-recovery-system.md

## Objective

Replace the prompt-hint-based stuck recovery with a deterministic `RecoveryController` that escalates through concrete strategies before invoking the LLM.

## Why this milestone exists

Current recovery: when stuck_score exceeds threshold, text hints like "try a different direction" are injected into the LLM prompt. The LLM then picks from the same candidate set with slightly different context. This is unreliable — the LLM may repeat the same failing action.

Recovery should be deterministic: try alternate path → interact nearby → backtrack → random walk → THEN ask LLM.

## Scope

- Create `RecoveryController` with 4 escalation tiers
- Simplify `stuck_detector.py` to score-only (no hint generation)
- Remove stuck hint injection from `context_manager.py`
- Wire recovery into the mode dispatcher (overrides normal controller when stuck)

## Non-goals

- Learning from stuck patterns (ML-based recovery)
- Recording stuck locations for future avoidance (can be added later)
- Changing stuck score thresholds

## Current code to inspect first

- `src/pokemon_agent/agent/stuck_detector.py`: full file
- `src/pokemon_agent/agent/context_manager.py`: stuck_warning section building
- `src/pokemon_agent/agent/engine.py`: stuck recovery integration

## Concrete code changes

### RecoveryController (`controllers/recovery.py`)

```python
class RecoveryController:
    def __init__(self, navigator: Navigator, objective_mgr: ObjectiveManager):
        self._navigator = navigator
        self._objective_mgr = objective_mgr
        self._tier = 0
        self._tier_attempts = 0

    def step(self, state, context) -> ActionDecision:
        if context.stuck_score < 4:
            self._tier = 0
            return None  # not stuck, let normal controller handle

        self._tier = self._compute_tier(context.stuck_score)

        if self._tier == 1:
            return self._try_alternate_path(state, context)
        elif self._tier == 2:
            return self._interact_nearby(state)
        elif self._tier == 3:
            return self._random_walk(state)
        else:  # tier 4
            return self._llm_recovery(state, context)
```

### Simplified stuck_detector.py

Reduce to ~60 lines:
- Track stuck_score (increment/decrement)
- Track oscillation (A→B→A→B detection)
- No hint generation
- No recovery_hint field
- No loop_signature (move to recovery controller if needed)

### Remove from context_manager.py

Delete the `stuck_warning` section entirely. Recovery is handled by the controller, not by prompting.

## Interfaces and contracts

### RecoveryController
- Returns None when not stuck (below threshold) — dispatcher falls through to normal controller
- Returns ActionDecision when stuck — overrides normal controller
- Resets on progress (any non-no_effect turn)
- Tier escalation: 1→2→3→4 based on consecutive no-effect turns

### Integration with dispatcher
The dispatcher checks recovery first:
```python
recovery_action = self._recovery.step(state, context)
if recovery_action is not None:
    return recovery_action
return self._controllers[state.mode].step(state, context)
```

## Data model changes

### Simplified: StuckState
Remove: recovery_hint, loop_signature, recent_failed_actions
Keep: score, oscillating, map_oscillating, steps_since_progress

## Migration plan

1. Create RecoveryController with tests
2. Simplify stuck_detector
3. Wire recovery into dispatcher
4. Remove stuck hints from context_manager
5. Verify stuck scenarios resolve faster

## Tests and evaluation

- Stuck score 4: tries alternate path
- Stuck score 6: interacts nearby
- Stuck score 8: random walk
- Stuck score 10: LLM escalation
- Progress resets all tiers
- Map change resets stuck score

## Acceptance criteria

1. Deterministic recovery tries 3 strategies before LLM
2. stuck_detector.py under 60 lines
3. No recovery hints in LLM prompts
4. Stuck loops resolve within 10 turns in common scenarios
5. All tests pass

## Rollback / fallback notes

If recovery controller makes things worse, disable it (let dispatcher skip recovery check) and re-enable stuck hints in context_manager.

## Ordered implementation checklist

1. **Step 6.1: Create RecoveryController**
   - Action: Implement 4-tier escalation
   - Files: `controllers/recovery.py`
   - Dependencies: Phase 3 (Navigator)
   - Done: all tiers produce actions
   - Tests: `test_recovery_controller.py`

2. **Step 6.2: Simplify stuck_detector**
   - Action: Remove hint generation, reduce to score tracking
   - Files: `agent/stuck_detector.py`
   - Dependencies: 6.1
   - Done: under 60 lines
   - Tests: score increment/decrement tests

3. **Step 6.3: Wire into dispatcher**
   - Action: Add recovery check before mode dispatch
   - Files: `agent/mode_dispatcher.py`
   - Dependencies: 6.1
   - Done: recovery overrides normal controller when stuck
   - Tests: integration test

4. **Step 6.4: Remove stuck hints from context_manager**
   - Action: Delete stuck_warning section
   - Files: `agent/context_manager.py`
   - Dependencies: 6.3
   - Done: no stuck hints in prompts
   - Tests: prompt content tests
