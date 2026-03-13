# Milestone 06 — OpenRouter LLM integration

## Goal
Connect the engine to OpenRouter with a strict action schema.

## Outcomes
- OpenRouter client works
- prompt builder works
- JSON action parsing works
- invalid responses are handled safely

## Required work
- configurable model via env/config
- request timeout/retry behavior
- redacted logging
- strict action validation
- fallback if model output is malformed

## Definition of done
- in mock mode, the engine can call OpenRouter and receive a validated action
- prompt contains current state, goals, recent events, relevant facts, and stuck warning if present
