# CLAUDE.md

## Project purpose
- LLM-driven agent that plays Pokémon Red autonomously via the PyBoy emulator
- Uses OpenRouter (configurable model) to plan actions and navigate the game world
- Runs a closed-loop: read game state → plan next action (LLM or deterministic executor) → send input → evaluate progress → update memory

---

## How to run

```bash
# Mock mode (no ROM needed, for development)
python -m pokemon_agent --mode mock

# Real emulator (requires ROM)
python -m pokemon_agent --mode pyboy --rom path/to/pokemon_red.gb

# Common flags
--planner {auto|llm|fallback}   # Planning strategy (default: auto)
--turns N                        # Stop after N turns
--continuous                     # Watch mode, re-run after completion
--session-dir DIR                # Resume from checkpoint
--fresh                          # Ignore existing checkpoint
--log-mode {dashboard|verbose|compact|quiet}
--debug-overlay                  # Visual debug output
--live-path-overlay              # Show pathfinding routes
```

**Environment setup:** copy `.env.example` → `.env` and set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`.

```bash
pip install -e ".[dev]"
```

---

## How to test

```bash
pytest                           # All tests
pytest tests/test_engine.py      # Engine turn loop tests
pytest tests/test_llm_client.py  # LLM client tests
pytest -v --tb=short             # Verbose with short tracebacks

# Lint / type check
ruff check src/
mypy src/
```

54 tests passing as of latest progress.md.

---

## Core entrypoints

| File | Role |
|------|------|
| `src/pokemon_agent/main.py` | CLI entry, builds emulator + runner, signal handling |
| `src/pokemon_agent/agent/engine.py` | `ClosedLoopRunner` — the main turn loop |
| `scripts/start_game.py` | Alternate launch script |
| `scripts/import_pret_world_graph.py` | Regenerates world graph JSON from Pret data |

---

## Architecture map

### Directory layout
```
src/pokemon_agent/
├── main.py              # CLI + orchestration
├── config.py            # Config dataclasses (from env)
├── agent/
│   ├── engine.py        # ClosedLoopRunner: run_turn(), _resolve_turn_plan()
│   ├── executor.py      # Deterministic task execution (no LLM)
│   ├── llm_client.py    # OpenRouter HTTP client
│   ├── navigation.py    # NavigationGrid BFS pathfinder
│   ├── world_map.py     # Connector discovery & inter-map BFS
│   ├── memory_manager.py # Event tracking, goal stack updates
│   ├── context_manager.py # Prompt building, token budgets
│   ├── progress.py      # State-diff progress classifier
│   ├── stuck_detector.py # Stuck score + recovery hints
│   ├── validator.py     # Action constraints + fallback
│   ├── menu_manager.py  # Menu navigation
│   └── battle_manager.py # Battle state + move selection
├── emulator/
│   ├── pyboy_adapter.py # PyBoy wrapper (real ROM)
│   ├── mock_adapter.py  # Deterministic mock for tests
│   └── pokemon_red_ram_map.py # Memory addresses
├── models/
│   ├── state.py         # StructuredGameState (Pydantic)
│   ├── action.py        # ActionType, ActionDecision, Task, StepResult
│   ├── memory.py        # MemoryState, GoalStack, LongTermKnowledge
│   └── planner.py       # CandidateNextStep, ObjectivePlanEnvelope
├── navigation/
│   └── world_graph.py   # Static world graph loader (from generated/)
├── data/
│   ├── walkthrough.py   # Story milestones + progression
│   ├── map_connections.py # Static map topology
│   └── map_names.py     # Map ID ↔ name
└── generated/           # Pre-computed world graph JSON (from Pret)
```

### Runtime loop (engine.py)
`run_turn(turn_index)`:
1. Read emulator state → `StructuredGameState`
2. `_resolve_turn_plan()` — up to 8 attempts:
   - Try executor first (if active task)
   - Fall back to `_plan_action()` → LLM or deterministic candidate selection
3. Validate action (`validator.py`)
4. Send to emulator
5. Compute progress diff
6. Update memory + stuck detector
7. Save checkpoint

### State model
`StructuredGameState` (`models/state.py`): map_name, map_id, x, y, facing, mode (OVERWORLD/MENU/TEXT/BATTLE/CUTSCENE/UNKNOWN), party, inventory, story_flags, badges, navigation snapshot (walkable/blocked tiles), NPC positions.

### Navigation
- **Intra-map**: `NavigationGrid` (BFS) in `agent/navigation.py` — temporary blockers with TTL
- **Inter-map**: `world_map.py` BFS over discovered connectors
- Connector discovery is incremental: SUSPECTED → CONFIRMED on actual traversal
- Blocked tile tracking: engine marks tiles blocked, executor clears on task start

### Planning
- **Deterministic**: `Executor` handles NAVIGATE_TO, NAVIGATE_ADJACENT, INTERACT, PRESS_BUTTON, ENTER_CONNECTOR, WALK_BOUNDARY
- **LLM**: `context_manager.py` builds prompt → `llm_client.py` → parse `CandidateNextStep` selection
- **Objective planner**: periodic LLM call to refresh high-level `ObjectivePlanEnvelope` (human_plan + internal_plan)
- Auto mode: executor runs first, LLM used when executor can't determine next step

### LLM integration
- `agent/llm_client.py`: OpenRouter HTTP client, JSON response format forced, retry + rate limiting
- `agent/context_manager.py`: builds system prompt + payload, enforces token budget (drops old action traces)
- Two prompt types: turn planner (select candidate_id) and objective planner (return plan JSON)

### Recovery / retry
- `stuck_detector.py`: stuck_score from repeated no-effect turns → recovery hints injected into prompt
- `validator.py`: max repeat limits, valid action filtering, fallback strategies
- Blocked tile TTL: temporary blockers expire automatically
- `_resolve_turn_plan()`: up to 8 attempts before giving up on a turn

---

## Important data models

| Model | Location | Key fields |
|-------|----------|------------|
| `StructuredGameState` | `models/state.py` | map_id, x, y, mode, party, navigation, npcs |
| `MemoryState` | `models/memory.py` | recent_events, goals (GoalStack), long_term (LongTermKnowledge) |
| `CandidateNextStep` | `models/planner.py` | id, type, target, why, priority, distance, advances_target |
| `ObjectivePlanEnvelope` | `models/planner.py` | human_plan, internal_plan, status |
| `Task` | `models/action.py` | task kind + target coords/button |
| `DiscoveredConnector` | `agent/world_map.py` | source/dest map+coords, activation_mode, status |
| `ActionTrace` | `agent/context_manager.py` | per-turn action, progress, events for prompt history |

World graph static data: `src/pokemon_agent/generated/*.json` (from Pret ROM data via `scripts/import_pret_world_graph.py`).

---

## Developer workflow notes

- **Mock mode** is the fastest dev loop — no ROM needed, deterministic states
- Checkpoints are JSON saved to `--session-dir`; use `--fresh` to reset
- `--log-mode verbose` or `--debug-overlay` for turn-by-turn inspection
- `--live-path-overlay` shows BFS pathfinding route in terminal
- `progress.md` tracks current working state and recent fixes — read it first when resuming
- Tests use mock emulator; real PyBoy integration not exercised in test suite
- To regenerate world graph: `python scripts/import_pret_world_graph.py`

---

## Constraints and gotchas

- Player coordinates: raw_y requires normalization (recently fixed bug — see progress.md)
- LLM gate: only fires after first controllable state (Red's House 2F) — bootstrap uses deterministic actions
- Connector activation modes vary: step_on vs push vs interact — wrong mode causes stuck loop
- Pathfinding BFS is intra-map only; inter-map requires world_map connector BFS
- Blocked tile TTL must expire naturally; forcibly clearing can cause re-blocking
- Token budget is enforced by dropping old action traces — long runs lose early history
- OpenRouter model is fully configurable; default in .env.example is `openai/gpt-5-mini` but any OpenRouter model works
- PyBoy dependency: requires a legally obtained Pokémon Red ROM (not included)

---

## External references

- [PyBoy](https://github.com/Baekalfen/PyBoy) — Game Boy emulator Python library
- [pret/pokered](https://github.com/pret/pokered) — Pokémon Red disassembly (source of world graph / RAM map data)
- [OpenRouter](https://openrouter.ai) — LLM API gateway used for model calls

---

## Unknowns / verify later

- Whether `scripts/start_game.py` adds anything over `python -m pokemon_agent` (not read in detail)
- Exact schema of generated world graph JSON in `src/pokemon_agent/generated/`
- Whether battle_manager.py and menu_manager.py are fully integrated or partially stubbed
- Token budget numbers (what the actual limits are — check `context_manager.py` constants)
- Whether `--continuous` watch mode auto-resumes from checkpoint or always restarts
- Integration test coverage for real PyBoy adapter vs mock
