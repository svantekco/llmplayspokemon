# Milestone 04 — Progress detection and stuck detection

## Goal
Compute whether actions are actually changing state, and detect loops early.

## Outcomes
- state diff engine exists
- progress classifications exist
- stuck score exists
- repeated no-op loops can be detected

## Required work
Classify action results into:
- no_effect
- movement_success
- interaction_success
- partial_progress
- major_progress
- regression
- unknown

Detect:
- repeated same-state loops
- repeated action with no effect
- oscillation between states
- too many steps without progress

## Definition of done
- mock scenarios can trigger and verify stuck detection
- loop behavior becomes visible in logs
