# Build Engine Master Prompt

Use this when asking a coding model to implement the project.

## Prompt

Build a local autonomous agent engine for “LLM plays Pokémon Red”.

The system should be designed so that an LLM does NOT directly control the emulator blindly. Instead, create a structured engine that reads game state, maintains memory, tracks progress, detects stuck behavior, calls an LLM through OpenRouter, validates actions, and then executes actions against an emulator adapter.

Use Python unless there is a very strong reason not to.

### Core goals
- read structured game state from an emulator adapter
- maintain short-term, mid-term, and long-term context
- send compact prompts to an LLM via OpenRouter
- receive a strictly structured action
- validate the action
- execute it
- measure whether real progress happened
- update memory only when meaningful events occur
- avoid repeated loops and dead behavior

### Implementation order
1. core models
2. mock emulator
3. PyBoy bootstrap
4. structured state extraction pipeline
5. action executor
6. progress and stuck logic
7. memory and goals
8. prompt builder
9. OpenRouter client
10. closed-loop runner

### Strong constraints
- the engine owns truth and memory
- progress must be computed from state changes
- memory updates should be event-driven
- raw button history is low value compared to summarized transitions
- action output must be strict JSON
- do not over-engineer battle AI in v1
- do not use LangChain or heavy agent frameworks
