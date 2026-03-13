# Milestone 05 — Memory and hierarchical goals

## Goal
Give the engine continuity so the LLM never feels stateless.

## Outcomes
- short-term memory stores meaningful recent transitions
- goal stack exists
- long-term memory exists for durable discoveries
- event-driven memory updates work

## Required work
Implement:
- immediate state
- short-term event history
- mid-term goal stack
- long-term knowledge store

Goal stack should include:
- long_term_goal
- mid_term_goal
- short_term_goal
- current_strategy
- success_conditions

## Definition of done
- engine can print compact memory context for the next step
- goal updates can happen via deterministic rules before any LLM involvement
