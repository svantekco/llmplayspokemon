Original prompt: Check session pathfinding turn 114. Why does not pathfinding suggest a path around the blocked tile? Why does it immediatly try to walk into a blocked tile. Fix it.

Also: Can you add the suggested path in the debug_overlay?

- Investigated `.sessions/pathfinding/debug_overlay/turn_0114.png` and `turn_0115.png`; the saved run repeats a blocked first step toward the staircase instead of rerouting.
- Patched planner routing to keep a short-lived memory of failed movement blockers even when the executor is not currently active, so fresh replans avoid the just-failed tile.
- Added suggested-path capture on executor steps and rendered that path in the debug overlay frame/minimap plus a short path summary in the side panel.
- Updated the overlay to call out the immediate next tile more clearly so the first move is easier to read from debug captures.
- Follow-up investigation on the live checkpoint showed the collision snapshot was vertically misaligned: emulator movement proves `MOVE_RIGHT` succeeds and `MOVE_UP` fails from the saved state, so the extracted `y` coordinate was two tiles too large for navigation. Normalized extracted active-state `y` by `-2` and preserved `raw_y` in metadata.
- Post-fix verification on `.sessions/pathfinding`: state loads as `(3,4)` with `raw_y=6`, and the top candidate now starts with `MOVE_RIGHT` and path `[(4,4), (4,3), ...]`.
- Added regression coverage for recent failed-move blocker rerouting and for debug overlay path rendering.
- Verification: `PYTHONPATH=src python3 -m pytest tests/test_executor.py tests/test_navigation.py tests/test_engine.py tests/test_debug_overlay.py` -> 43 passed.
