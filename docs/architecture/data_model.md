# Core Data Model

## StructuredGameState
- map_name
- map_id
- x
- y
- facing
- mode
- menu_open
- text_box_open
- battle_state
- party
- inventory
- step
- metadata

## ActionDecision
- action
- repeat
- reason

## GoalStack
- long_term_goal
- mid_term_goal
- short_term_goal
- current_strategy
- success_conditions

## EventRecord
- type
- summary
- step
- metadata

## ProgressResult
- classification
- changed_fields
- newly_completed_subgoals
- notes

## StuckState
- score
- recent_failed_actions
- loop_signature
- recovery_hint
