# Architecture Overview

## High-level loop

```text
Emulator -> State Extractor -> Goal/Memory -> Prompt Builder -> OpenRouter LLM
                                                              |
                                                              v
                                                      Action Validator
                                                              |
                                                              v
                                                       Executor/Input
                                                              |
                                                              v
                                                   Progress + Stuck Detector
                                                              |
                                                              v
                                                     Memory / Goal Updates
```

## Design rules

- Emulator state is the source of truth.
- The LLM decides the next action, not the timing internals.
- Progress is computed from state diffs.
- Memory is layered.
- Stuck detection is mandatory.
- Early milestones should work with deterministic scripts and mocks before the LLM loop is trusted.

## Memory layers

### Immediate state
What is true right now.

### Short-term
Recent meaningful transitions, not raw input spam.

### Mid-term
Objective stack and current strategy.

### Long-term
Durable discoveries: routes, blockers, story flags, landmarks, learned recovery hints.
