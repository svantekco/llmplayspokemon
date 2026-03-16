# 02-deterministic-controllers.md

## Objective

Replace per-turn LLM candidate selection for dialogue, battle, menu, and cutscene modes with controllers that use deterministic execution. Battle gets **one LLM strategy call per encounter** (or per new opposing Pokemon in trainer battles) to set high-level strategy, then executes that strategy deterministically turn-by-turn.

## Why this milestone exists

These four game modes should not route every action through the LLM candidate pipeline. Dialogue and menus are fully deterministic. Battle benefits from one strategic LLM call (gym leaders, catch decisions) but the per-turn menu navigation is deterministic. This is the single highest-leverage change for reducing LLM dependency.

## Scope

- `DialogueController`: advance text, deterministic yes/no, flag unknown choices
- `BattleController`: one LLM call per encounter for strategy, then deterministic execution
- `MenuController`: cursor-based menu navigation to target
- `CutsceneController`: PRESS_A or wait
- Remove candidate building for these modes from engine.py

## Non-goals

- Full battle simulator (no damage prediction, no speed calc)
- Full menu state machine (basic cursor navigation is sufficient)
- Story-branching dialogue (flag for LLM, don't solve now)
- Shop/heal automation (deferred to later; close menu for now)

## Current code to inspect first

- `src/pokemon_agent/agent/battle_manager.py`: `build_candidates()`, `_move_score()`, `_has_type_disadvantage()`
- `src/pokemon_agent/agent/menu_manager.py`: `build_candidates()`, `_detect_menu()`, `_menu_action()`
- `src/pokemon_agent/agent/engine.py`: `_build_candidate_steps()` branches for TEXT, MENU, BATTLE
- `src/pokemon_agent/agent/validator.py`: `fallback()` for mode-specific defaults

## Concrete code changes

### DialogueController (`controllers/dialogue.py`)

Core logic:
```python
def step(self, state, context):
    if state.text_box_open:
        if self._is_yes_no_prompt(state):
            choice = self._deterministic_choice(state)
            if choice is not None:
                return choice  # ActionDecision for yes or no
            # Unknown choice — default to yes (safe for most game prompts)
            return ActionDecision(action=ActionType.PRESS_A, reason="advance dialogue (default yes)")
        return ActionDecision(action=ActionType.PRESS_A, reason="advance text")
    # Text mode but no text box — transitional, press A
    return ActionDecision(action=ActionType.PRESS_A, reason="clear text transition")
```

Yes/No deterministic rules (extracted from existing engine.py):
- Keywords "heal", "take", "accept", "want", "receive" → YES
- Keywords "save", "quit", "cancel", "give up" → NO
- Pokemon Center "restore your Pokemon" → YES
- Shop "is there anything" → NO (exit shop)

### BattleController (`controllers/battle.py`)

**Two-layer design: LLM strategy + deterministic execution.**

#### Layer 1: Strategy (one LLM call per encounter / per new opposing Pokemon)

When a battle starts or a new opposing Pokemon appears, the controller calls the LLM once to get a `BattleStrategy`:

```python
@dataclass
class BattleStrategy:
    lead_pokemon: str | None          # switch to this if not already active
    preferred_moves: list[str]        # ordered move preferences ["THUNDERBOLT", "QUICK_ATTACK"]
    switch_threshold_hp_pct: int      # switch if HP drops below this %
    switch_target: str | None         # who to switch to if threshold hit
    use_items: bool                   # whether to heal with potions
    should_catch: bool                # for wild encounters: try to catch?
    should_run: bool                  # for trivial wilds: just run
    notes: str                        # LLM reasoning (for debug logging)
```

The LLM prompt includes:
- Current party (species, level, HP, moves with PP, types)
- Current inventory (balls, potions)
- Opposing Pokemon (species, level, estimated HP %, types)
- Whether this is wild, trainer, gym leader, or rival
- Current roster gaps ("no Water type in party" etc.)

**Trivial-wild skip:** If the encounter is a wild Pokemon with level ≤ (player avg level - 5) and the player has no catch interest (species already in party, no balls, etc.), skip the LLM call entirely and use a hardcoded "run or one-shot with best move" strategy.

**Catch decision:** The LLM evaluates whether to catch based on:
- Is this species already in the party?
- Does the party have coverage gaps this species would fill?
- Does the player have Poke Balls?
- Is the opponent's HP low enough to catch reliably?
- Is this a rare/useful species for upcoming challenges?

#### Layer 2: Execution (fully deterministic, every turn)

```python
def step(self, state, context):
    battle = state.battle_state
    if battle is None:
        return PlanningResult(action=ActionDecision(action=ActionType.PRESS_A, reason="no battle context"))

    # On new battle or new opponent: get strategy (may call LLM)
    if self._needs_new_strategy(battle):
        self._strategy = self._get_strategy(state, context)

    # Determine target action from strategy
    target = self._resolve_target_action(battle, self._strategy)

    # Navigate battle menu deterministically to execute target
    return PlanningResult(action=self._navigate_menu_to(battle, target))
```

Target action resolution (deterministic, from strategy):
- If `strategy.should_run` → RUN
- If `strategy.should_catch` and have balls → BAG → ball
- If `strategy.lead_pokemon` != active Pokemon → POKEMON → switch
- If HP < `strategy.switch_threshold_hp_pct` and `strategy.use_items` and have potion → BAG → potion
- If HP < `strategy.switch_threshold_hp_pct` and `strategy.switch_target` → POKEMON → switch
- Otherwise → FIGHT → first available move from `strategy.preferred_moves` with PP > 0
- Fallback (no PP left, no strategy match) → best move by `power × type × STAB`

#### LLM call budget

| Scenario | LLM calls | Current system |
|---|---|---|
| Trivial wild (level gap > 5) | 0 | 3-8 |
| Wild worth evaluating | 1 | 3-8 |
| Trainer (3 Pokemon) | up to 3 | 10-25 |
| Gym leader (4-6 Pokemon) | up to 6 | 15-40+ |

#### Fallback when LLM unavailable

If `llm_client` is None (fallback planner mode), skip the LLM call and use pure heuristic:
- Score = power × type_effectiveness × STAB_bonus
- If best score == 0 (no damaging moves): use first move with PP
- Wild + overleveled: RUN
- HP < 25% + potion available: heal
- All fainted: wait for whiteout

This keeps the controller functional without LLM access.

### MenuController (`controllers/menu.py`)

Core logic:
```python
def step(self, state, context):
    # Default: close menu (PRESS_B)
    # If we have a menu target (e.g., use HM, heal), navigate to it
    # Otherwise: exit menu to return to overworld
    return ActionDecision(action=ActionType.PRESS_B, reason="close menu")
```

For now, the menu controller's primary job is to GET OUT of menus. Navigating into menus for specific purposes (HM use, item use) can be added incrementally.

### CutsceneController (`controllers/cutscene.py`)

```python
def step(self, state, context):
    return ActionDecision(action=ActionType.PRESS_A, reason="advance cutscene")
```

## Interfaces and contracts

### DialogueController
- Input: `StructuredGameState` with `text_box_open=True`
- Output: `ActionDecision` (PRESS_A, or directional for yes/no cursor)
- Contract: always returns an action; never calls LLM
- Stateless (no internal state needed)

### BattleController
- Input: `StructuredGameState` with `battle_state is not None`; optional `llm_client` for strategy calls
- Output: `PlanningResult` (direction to navigate battle menu, or PRESS_A to confirm)
- Contract: one LLM call per encounter/per new opposing Pokemon for strategy; all per-turn actions deterministic
- State: `BattleStrategy` (set once per encounter, updated on new opponent), current submenu tracking
- LLM-free fallback: pure heuristic strategy when `llm_client` is None

### MenuController
- Input: `StructuredGameState` with `menu_open=True`
- Output: `ActionDecision` (PRESS_B to close, or navigation to target)
- Contract: default behavior is to close the menu
- Stateless for now

### CutsceneController
- Input: `StructuredGameState` with `mode=CUTSCENE`
- Output: `ActionDecision` (PRESS_A)
- Stateless, trivial

## Data model changes

Controllers use existing `ActionDecision` and `StructuredGameState`.

### New: `BattleStrategy`
```python
@dataclass
class BattleStrategy:
    lead_pokemon: str | None
    preferred_moves: list[str]
    switch_threshold_hp_pct: int
    switch_target: str | None
    use_items: bool
    should_catch: bool
    should_run: bool
    notes: str
```
- Owner: `BattleController` (set via LLM or heuristic fallback)
- Lifetime: one per encounter, refreshed when opposing Pokemon changes
- Not persisted to checkpoints (battles don't span checkpoints)

### Internal: `BattlePhase`
```python
class BattlePhase(Enum):
    NEED_STRATEGY = "need_strategy"    # awaiting LLM or heuristic strategy
    SELECT_ACTION = "select_action"    # main menu
    SELECT_MOVE = "select_move"        # fight submenu
    SELECT_POKEMON = "select_pokemon"  # switch submenu
    SELECT_ITEM = "select_item"        # bag submenu
    AWAITING_RESULT = "awaiting"       # animation/result
```

## Migration plan

1. Create controller files (no behavioral change yet)
2. Write tests for each controller in isolation
3. Replace stub controllers from Phase 1 with real controllers one at a time
4. After each replacement, run full test suite
5. Remove corresponding `_build_candidate_steps()` branches from engine.py

Order: DialogueController first (simplest), then CutsceneController (trivial), then BattleController (most logic), then MenuController (close-menu default).

## Tests and evaluation

### DialogueController tests
- Text box open → returns PRESS_A
- Yes/no prompt with "heal" → returns yes action
- Yes/no prompt with "save game" → returns no action
- No text box (transitional) → returns PRESS_A

### BattleController tests
- Trivial wild (level gap > 5) → no LLM call, runs or one-shots
- Wild with catch interest → LLM called once, strategy includes should_catch
- Trainer battle → LLM called once for strategy, deterministic execution follows
- New opposing Pokemon → LLM called again for updated strategy
- LLM unavailable → falls back to heuristic (power × type × STAB)
- Strategy preferred_moves respected in order (skip moves with 0 PP)
- HP below switch_threshold → switches or heals per strategy
- Menu navigation: cursor at FIGHT, target FIGHT → PRESS_A
- Menu navigation: cursor at ITEM, target FIGHT → correct navigation

### MenuController tests
- Menu open → returns PRESS_B (close menu)
- (Future: menu target set → navigates to target)

### CutsceneController tests
- Any state → returns PRESS_A

### Existing tests
- All 54+ tests must continue to pass

## Risks and edge cases

- **Battle menu position may be None.** If RAM read fails, the controller can't navigate the menu. Fallback: PRESS_A (safe default — confirms current selection).
- **LLM strategy may be bad.** The LLM might recommend a move that doesn't exist or a switch to a fainted Pokemon. Mitigate: validate strategy against actual party/moves before executing; fall back to heuristic for invalid entries.
- **LLM latency in battle.** One call per encounter adds ~1-3 seconds. Acceptable since battles take many turns anyway.
- **"Trivial wild" threshold may be wrong.** Level gap of 5 might skip encounters that are worth catching. Mitigate: also check if species is not already in party and balls are available before skipping.
- **Yes/no keyword matching may miss edge cases.** Some dialogues don't contain expected keywords. Fallback: default to YES (most game prompts are positive — heal, accept, etc.).
- **Menu controller closing menus may interrupt intended menu use.** If the objective requires menu interaction (HM use), the close-menu default is wrong. Mitigate: check `context.objective` for menu-related goals.
- **Battle type chart may be incomplete.** Current `TYPE_EFFECTIVENESS` dict in battle_manager.py may miss matchups. Will be fixed properly in Phase 4 with ROM import.

## Acceptance criteria

1. DialogueController advances text without LLM in all text states
2. BattleController makes at most one LLM call per encounter (or per new opposing Pokemon); all per-turn actions are deterministic
3. BattleController works in pure-heuristic mode when LLM is unavailable
4. Trivial wild encounters skip the LLM call entirely
5. MenuController exits menus without LLM
6. CutsceneController advances cutscenes without LLM
7. No per-turn LLM calls occur for TEXT, MENU, or CUTSCENE modes
8. All existing tests pass
9. Each controller has its own unit test file with ≥ 5 test cases

## Rollback / fallback notes

Each controller is independent. If one controller introduces regressions:
- Swap it back to the stub (delegates to existing engine logic)
- No other controllers are affected
- The dispatcher handles the swap transparently

## Ordered implementation checklist

1. **Step 2.1: Create DialogueController**
   - Action: Implement text advancement + deterministic yes/no logic
   - Files: `controllers/dialogue.py`
   - Dependencies: Phase 1 complete
   - Done: handles all text states without LLM
   - Tests: `test_dialogue_controller.py` with 5+ cases

2. **Step 2.2: Create CutsceneController**
   - Action: Implement PRESS_A for cutscene states
   - Files: `controllers/cutscene.py`
   - Dependencies: Phase 1 complete
   - Done: handles cutscene mode
   - Tests: trivial test

3. **Step 2.3: Create BattleController**
   - Action: Implement two-layer battle controller: LLM strategy call (one per encounter/new opponent) + deterministic menu execution. Include trivial-wild skip (no LLM), catch decision support, and pure-heuristic fallback when LLM unavailable.
   - Files: `controllers/battle.py`
   - Dependencies: Phase 1 complete
   - Done: one LLM call sets `BattleStrategy`; per-turn step() is deterministic; works without LLM
   - Tests: `test_battle_controller.py` — trivial wild skip, strategy execution, LLM fallback, catch decision, new-opponent refresh

4. **Step 2.4: Create MenuController**
   - Action: Implement close-menu default with objective awareness
   - Files: `controllers/menu.py`
   - Dependencies: Phase 1 complete
   - Done: closes menus, respects context objectives
   - Tests: `test_menu_controller.py`

5. **Step 2.5: Wire controllers into dispatcher**
   - Action: Replace stubs with real controllers in engine's dispatcher initialization
   - Files: `agent/engine.py`
   - Dependencies: 2.1-2.4
   - Done: all non-overworld modes use real controllers
   - Tests: full test suite passes

6. **Step 2.6: Remove candidate branches for handled modes**
   - Action: Delete TEXT/BATTLE/MENU/CUTSCENE branches from `_build_candidate_steps()`
   - Files: `agent/engine.py`
   - Dependencies: 2.5
   - Done: `_build_candidate_steps()` only handles OVERWORLD
   - Tests: full test suite passes
