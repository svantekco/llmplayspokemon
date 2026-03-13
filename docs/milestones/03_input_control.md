# Milestone 03 — Nail controlling the game

## Goal
Make control reliable enough that deterministic scripts can move through menus or overworld without an LLM.

## Outcomes
- action abstraction exists
- repeat counts are supported
- timing/frame advancement is handled centrally
- deterministic test scripts can press A/B/Start/direction buttons
- action execution results are logged

## Required work
- implement executor logic for actions:
  - MOVE_UP
  - MOVE_DOWN
  - MOVE_LEFT
  - MOVE_RIGHT
  - PRESS_A
  - PRESS_B
  - PRESS_START
- centralize debounce / frame timing assumptions
- add a demo script that performs a scripted sequence

## Guardrails
- do not let the future LLM own frame timing
- do not hide failed inputs
- keep input semantics explicit and testable

## Definition of done
- a scripted route or menu interaction can be replayed from code
- action execution is deterministic enough to debug
