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
- Investigated `.sessions/game` and reproduced that the planner did have a canonical stair candidate for `Red's House 2F -> Red's House 1F`, but executor handoff blocked immediately with `No path to (7, 1)` and the turn fell back to `PRESS_A`.
- Root cause: static warp synthesis was treating any boundary-coordinate warp as a boundary `push` connector. That misclassified the explicit staircase at `(7,1)` as an east-edge push exit, so the executor targeted the unreachable warp tile instead of a reachable approach tile.
- Patched static connector synthesis to reserve boundary-push semantics for `LAST_MAP`/`last_map_proximity` exits only, and narrowed the walkable-warp south-entry bias to actual south-edge exits so staircases can use the shortest adjacent approach.
- Verification: `PYTHONPATH=src python3 -m pytest tests/test_engine.py tests/test_executor.py` -> 53 passed.
- Verification: loading `.sessions/game` now yields `planner=auto_candidate`, `candidate=static_warp_reds_house_1f_7_1`, `action=MOVE_UP`, and suggested path `[(4, 5), (4, 4), (4, 3), (4, 2), (5, 2), (6, 2), (7, 2)]` instead of fallback `PRESS_A`.
- Tightened the opening-force LLM gate so it now waits for the first controllable `Red's House 2F` turn during `get_starter` instead of forcing on any early-game overworld state.
- The new trigger fires when the player is on `Red's House 2F` and either just exited bootstrap or the session is starting there before that map has been recorded in known locations.
- Added regression coverage that bootstrap still skips the LLM, `Mock Town` does not force the opening call, and the first controllable `Red's House 2F` turn does force the LLM.
- Verification: `PYTHONPATH=src python3 -m pytest tests/test_engine.py tests/test_executor.py` -> 54 passed.
