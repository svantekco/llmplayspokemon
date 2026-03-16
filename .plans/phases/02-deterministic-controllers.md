# 02-deterministic-controllers.md

## Objective

Replace LLM-based candidate selection for dialogue, battle, menu, and cutscene modes with deterministic controllers that always produce the correct action without an LLM call.

## Why this milestone exists

These four game modes have known-correct deterministic behavior. Routing them through the candidate → LLM pipeline wastes API tokens, adds latency, and introduces failure modes. This is the single highest-leverage change for reducing LLM dependency.

## Scope

- `DialogueController`: advance text, deterministic yes/no, flag unknown choices
- `BattleController`: move selection via type chart + power, switch/run heuristics
- `MenuController`: cursor-based menu navigation to target
- `CutsceneController`: PRESS_A or wait
- Remove candidate building for these modes from engine.py

## Non-goals

- Perfect battle strategy (good enough heuristic > optimal play)
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

Core logic (extracted and simplified from `battle_manager.py`):
```python
def step(self, state, context):
    battle = state.battle_state
    if battle is None:
        return ActionDecision(action=ActionType.PRESS_A, reason="no battle context")

    # Navigate to correct submenu first
    menu_nav = self._navigate_to_submenu(battle, target_submenu)
    if menu_nav is not None:
        return menu_nav

    # In fight submenu: select best move
    if self._in_fight_menu(battle):
        move_idx = self._best_move_index(battle)
        return self._navigate_to_move(battle, move_idx)

    # Main menu: select FIGHT
    return self._navigate_to_fight(battle)
```

Move selection:
- Score = power × type_effectiveness × STAB_bonus
- If best score == 0 (no damaging moves): use first move with PP
- Wild + overleveled: prefer RUN
- HP < 25% + potion available: prefer ITEM (heal)
- All fainted: wait for whiteout

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
- Input: `StructuredGameState` with `battle_state is not None`
- Output: `ActionDecision` (direction to navigate battle menu, or PRESS_A to confirm)
- Contract: tracks current battle submenu position across calls
- State: current submenu (MAIN/FIGHT/POKEMON/BAG), target action

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

No new models. Controllers use existing `ActionDecision` and `StructuredGameState`.

The `BattleController` may define an internal `BattlePhase` enum:
```python
class BattlePhase(Enum):
    SELECT_ACTION = "select_action"  # main menu
    SELECT_MOVE = "select_move"      # fight submenu
    SELECT_POKEMON = "select_pokemon" # switch submenu
    AWAITING_RESULT = "awaiting"     # animation/result
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
- Wild battle, 4 moves → selects highest-scoring move
- Type advantage matchup → correct effectiveness multiplier
- Low HP + potion → selects heal item
- Wild + overleveled → selects RUN
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
- **Yes/no keyword matching may miss edge cases.** Some dialogues don't contain expected keywords. Fallback: default to YES (most game prompts are positive — heal, accept, etc.).
- **Menu controller closing menus may interrupt intended menu use.** If the objective requires menu interaction (HM use), the close-menu default is wrong. Mitigate: check `context.objective` for menu-related goals.
- **Battle type chart may be incomplete.** Current `TYPE_EFFECTIVENESS` dict in battle_manager.py may miss matchups. Will be fixed properly in Phase 4 with ROM import.

## Acceptance criteria

1. DialogueController advances text without LLM in all text states
2. BattleController selects moves without LLM in all battle states
3. MenuController exits menus without LLM
4. CutsceneController advances cutscenes without LLM
5. No LLM calls occur for TEXT, BATTLE, MENU, or CUTSCENE modes
6. All existing tests pass
7. Each controller has its own unit test file with ≥ 5 test cases

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
   - Action: Extract battle_manager logic into controller, add submenu navigation
   - Files: `controllers/battle.py`
   - Dependencies: Phase 1 complete
   - Done: selects moves, handles switching, run, items
   - Tests: `test_battle_controller.py` with type chart coverage

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
