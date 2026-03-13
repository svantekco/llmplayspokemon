# Pokémon LLM Agent Starter Kit

A starter repo for building an autonomous **LLM plays Pokémon Red** engine.

This kit is organized around **large milestones** rather than tiny tasks. The first milestones focus on the part that matters most:

1. installing and connecting a Game Boy emulator
2. proving reliable read access to game state
3. proving reliable control/input
4. only then layering in the LLM engine, memory, and progress logic

## Recommended stack

- **Language:** Python
- **Initial emulator target:** **PyBoy**
- **LLM gateway:** **OpenRouter**
- **Core libraries:** pydantic, httpx, pytest

## Why this order

Most agent projects fail because they start with prompts before they have deterministic control.

For Pokémon Red, the right order is:

- emulator integration
- state extraction
- input/control
- deterministic test loops
- progress detection
- memory and goals
- LLM planning

## Included in this starter kit

- milestone plan split into larger phases
- architecture notes
- prompt pack for coding agents
- Python scaffold for the agent engine
- mock and PyBoy emulator adapters
- starter schemas for state, action, memory, events
- `.env.example`
- test stubs and utility scripts

## Suggested milestone order

Read these in order:

- `docs/milestones/00_repo_setup.md`
- `docs/milestones/01_emulator_bootstrap.md`
- `docs/milestones/02_state_extraction.md`
- `docs/milestones/03_input_control.md`
- `docs/milestones/04_progress_and_stuck.md`
- `docs/milestones/05_memory_and_goals.md`
- `docs/milestones/06_llm_openrouter_integration.md`
- `docs/milestones/07_closed_loop_agent.md`
- `docs/milestones/08_hardening_and_eval.md`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .[dev]
cp .env.example .env
```

## Run mock mode

```bash
python -m pokemon_agent.main --mode mock
```

The default terminal view is a live dashboard with:

- current game state at the top
- recent turns in the middle
- the latest LLM request/response details at the bottom

## Run PyBoy mode later

```bash
python -m pokemon_agent.main --mode pyboy --rom path/to/PokemonRed.gb
```

## Start or resume a local session

This launcher starts PyBoy with `game.gb` by default and automatically resumes from an existing checkpoint if the session directory already contains one.

```bash
python3 scripts/start_game.py --session-dir .sessions/default
```

Start a fresh session instead of resuming:

```bash
python3 scripts/start_game.py --session-dir .sessions/default --fresh
```

Use mock mode for quick smoke tests:

```bash
python3 scripts/start_game.py --mode mock --session-dir .sessions/mock --turns 2
```

## Important note

The provided PyBoy adapter is intentionally a thin placeholder plus basic hooks. The goal is to give your coding model a clean place to implement emulator work during the early milestones.
